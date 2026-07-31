import Foundation

/// Bridge to the Python engine.
///
/// The app reimplements nothing: `scriba` does the work, this class launches it,
/// reads its output to follow where it has got to, and re-reads the state once it is done.
/// That is deliberate. The engine has to stay usable on its own from the terminal,
/// and an app that duplicates the logic starts drifting the day after.
/// The part of a run that changes several times a second.
///
/// Kept apart from the Engine on purpose. When these lived on the Engine, every
/// publish rebuilt every view observing it, which was the whole window: the
/// sidebar list, the toolbar and whatever was in the detail pane, four times a
/// second for the length of a transcription. Only the two views that draw
/// progress observe this one.
@MainActor
final class RunState: ObservableObject {
    @Published var phase: Phase = .idle
    @Published var lastLine = ""
    @Published var progress: Double?

    func reset(to phase: Phase) {
        self.phase = phase
    }
}

@MainActor
final class Engine: ObservableObject {

    /// Live progress, observed only where it is drawn.
    let live = RunState()

    /// Where the run has got to. Published because the queue and the panels
    /// switch on it, and it changes seven times in a run rather than hundreds.
    @Published var phase: Phase = .idle

    /// The newest line worth showing, and how far the current stage has got.
    ///
    /// The whole log used to be @Published and appended to per line. whisper
    /// prints a progress percentage continuously, so every one of those lines
    /// republished the entire array and SwiftUI rebuilt everything observing this
    /// object, sidebar included. The window went sticky for the length of a
    /// transcription, which is exactly as long as somebody most wants to use it.
    /// Kept for the failure message. Not published: nothing draws from it while
    /// the job runs, and republishing it was the whole problem.
    private(set) var log: [String] = []
    private var lastPublish = Date.distantPast
    private var residue = ""
    @Published var info: JobInfo?
    @Published var speakers: [Speaker] = []
    @Published var errorText: String?
    @Published var isRunning = false
    /// True while the state is being read in the background, so the interface can
    /// say it is busy instead of appearing frozen.
    @Published var isLoadingState = false

    private var process: Process?

    /// Set while a stop the user asked for is in flight.
    ///
    /// Terminating the child makes it exit non-zero, which is indistinguishable
    /// from a crash unless somebody writes it down. Without this the Stop button
    /// worked and then apologised for working, with an alert quoting the last six
    /// lines of a log nobody needed.
    private(set) var wasCancelled = false

    /// One line saying why the last run stopped, for a list that has room for one
    /// line. The alert carries the full text and is dismissed once; this stays.
    var failureSummary: String {
        guard let text = errorText, !text.isEmpty else { return "did not finish" }
        let firstLine = text.split(separator: "\n").first.map(String.init) ?? text
        let trimmed = firstLine.trimmingCharacters(in: .whitespaces)
        return trimmed.count > 90 ? String(trimmed.prefix(88)) + "…" : trimmed
    }

    // MARK: - configuration

    /// Where the Python interpreter with scriba installed lives.
    ///
    /// The guess is the conda environment holding whisperX 3.8.6 and pyannote 4, the
    /// combination that puts diarization on Metal. Settings overrides it, and the
    /// error path names the path it tried, because "nothing happens when I press
    /// Transcribe" is the least useful thing an app can do.
    static var pythonPath: String {
        get {
            UserDefaults.standard.string(forKey: "pythonPath")
                ?? "/opt/homebrew/Caskroom/miniforge/base/envs/whisperx4/bin/python"
        }
        set { UserDefaults.standard.set(newValue, forKey: "pythonPath") }
    }

    /// Folder of the Python package (only needed when scriba is not pip-installed).
    static var enginePath: String {
        get {
            UserDefaults.standard.string(forKey: "enginePath")
                ?? (NSHomeDirectory() as NSString).appendingPathComponent("dev/scriba")
        }
        set { UserDefaults.standard.set(newValue, forKey: "enginePath") }
    }

    static var isConfigured: Bool {
        FileManager.default.isExecutableFile(atPath: pythonPath)
    }

    /// Whether the working directory we are about to hand the process exists.
    ///
    /// It does not have to hold the package: a pip install puts scriba on the
    /// path and the folder is then only a place to start from. It does have to
    /// exist, because Process throws before running anything if it does not, and
    /// the error that surfaces then names the interpreter and not the folder.
    static var isEngineFolderUsable: Bool {
        var isDir: ObjCBool = false
        let there = FileManager.default.fileExists(atPath: enginePath, isDirectory: &isDir)
        return there && isDir.boolValue
    }

    /// Mirrors ERR_NO_TOKEN in scriba/pipeline.py. The one string that genuinely
    /// crosses the Python/Swift boundary, so it is spelled out in one place on
    /// each side rather than buried inside a condition.
    static let errNoToken = "[scriba:error:hf-token]"

    // MARK: - execution

    /// How a run ended. A queue needs all three: it moves on after the first two
    /// and stops after the third, and a stop the user asked for is not a failure.
    enum Outcome { case finished, failed, cancelled }

    /// `onFinish` receives the outcome, so a queue can move on to the next
    /// recording. Without it the caller has to poll `isRunning`, and a failure
    /// looks exactly like a success that ran quickly.
    func run(file: URL, language: String, minSpeakers: Int?, maxSpeakers: Int?,
             onFinish: ((Outcome) -> Void)? = nil) {
        guard !isRunning else { return }
        var args = ["-m", "scriba.cli", "run", file.path, "--lang", language]
        if let m = minSpeakers { args += ["--min-speakers", String(m)] }
        if let m = maxSpeakers { args += ["--max-speakers", String(m)] }
        launch(args, on: file, startingPhase: .preparing, onFinish: onFinish)
    }

    /// True while `scriba name` is running, as opposed to a transcription.
    ///
    /// The two are different jobs of work and the interface said the same thing
    /// about both: pressing Save showed a progress strip with an empty filename
    /// walking through "preparing the audio", which is not what saving a name
    /// does and not how long it takes.
    @Published private(set) var isNaming = false

    func applyNames(file: URL, mapping: [String: String], enroll: Bool,
                    onFinish: ((Outcome) -> Void)? = nil) {
        guard !isRunning else { return }
        var args = ["-m", "scriba.cli", "name", file.path]
        args += mapping.filter { !$0.value.isEmpty }.map { "\($0.key)=\($0.value)" }
        if !enroll { args.append("--no-enroll") }
        isNaming = true
        launch(args, on: file, startingPhase: .identifying) { [weak self] outcome in
            self?.isNaming = false
            onFinish?(outcome)
        }
    }

    private func launch(_ args: [String], on file: URL, startingPhase: Phase,
                        onFinish: ((Outcome) -> Void)? = nil) {
        isRunning = true
        wasCancelled = false
        errorText = nil
        log.removeAll()
        lastPublish = .distantPast
        residue = ""
        phase = startingPhase
        live.reset(to: startingPhase)

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: Self.pythonPath)
        proc.arguments = args
        // Fall back to the home directory rather than a folder that is not there.
        // The default points at a clone, and somebody who installed with pip has
        // no clone, so the run failed with a message about the interpreter.
        proc.currentDirectoryURL = URL(fileURLWithPath:
            Self.isEngineFolderUsable ? Self.enginePath : NSHomeDirectory())

        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = Self.enginePath
        env["PYTHONUNBUFFERED"] = "1"
        // Each audio library opens its own thread pool; without this cap they fight
        // over the same cores and end up slower, not faster.
        // An app launched from the Finder inherits a PATH with almost nothing on
        // it, and the engine shells out to ffmpeg and ffprobe. Without this the
        // first stage failed with "has no audio track", which is a confident,
        // specific and wrong thing to say about a file that plays fine, and it
        // only happened for people who did not start the app from a terminal.
        let extraPath = [URL(fileURLWithPath: Self.pythonPath).deletingLastPathComponent().path,
                         "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
        env["PATH"] = (extraPath + [env["PATH"] ?? ""]).joined(separator: ":")
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["OMP_NUM_THREADS"] = String(max(ProcessInfo.processInfo.activeProcessorCount / 2, 1))
        proc.environment = env

        let outPipe = Pipe(), errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        // Both pipes, and both for two separate reasons.
        //
        // Progress: the engine prints its phase lines on stdout. Both rich's Console
        // and plain print() go there, and stderr stays empty on a clean run. Reading
        // only stderr meant Phase.from never saw a single line, and the progress panel
        // sat on its first phase for the whole job.
        //
        // Survival: an unread pipe fills up. The OS buffer is 64 KB, whisper's own
        // progress output is far more than that, and a child process whose stdout is
        // full blocks forever. Draining stdout is not about showing it. It is what
        // keeps a long transcription from hanging.
        let onData: @Sendable (FileHandle) -> Void = { [weak self] handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty, let text = String(data: chunk, encoding: .utf8) else { return }
            Task { @MainActor in self?.ingest(text) }
        }
        outPipe.fileHandleForReading.readabilityHandler = onData
        errPipe.fileHandleForReading.readabilityHandler = onData

        proc.terminationHandler = { [weak self] p in
            // Whatever arrived after the last callback. The reason a run failed is
            // usually in the final few lines, and they were being dropped.
            let tail = (try? outPipe.fileHandleForReading.readToEnd()).flatMap { $0 }
            let errTail = (try? errPipe.fileHandleForReading.readToEnd()).flatMap { $0 }
            Task { @MainActor in
                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                for data in [tail, errTail] {
                    if let data, let text = String(data: data, encoding: .utf8) {
                        self?.ingest(text + "\n")
                    }
                }
                // Stop, then start again straight away, and this handler belongs
                // to the process that was killed. Letting it run marked the new
                // run as failed and cleared the state out from under it.
                guard self?.process == nil || self?.process === p else { return }
                self?.isRunning = false
                self?.process = nil
                if self?.wasCancelled == true {
                    self?.wasCancelled = false
                    self?.phase = .idle
                    self?.live.reset(to: .idle)
                    self?.errorText = nil
                    onFinish?(.cancelled)
                    return
                }
                let ok = p.terminationStatus == 0
                if ok {
                    self?.phase = .done
                    self?.reload(file: file)
                } else {
                    self?.phase = .failed
                    // Always something. The old version tested for nil, and an
                    // empty log produced an empty string, so the alert came up
                    // with a title, a Close button and no text at all.
                    if (self?.errorText ?? "").isEmpty {
                        let tail = self?.log.suffix(6).joined(separator: "\n") ?? ""
                        self?.errorText = tail.isEmpty
                            ? "The engine exited with code \(p.terminationStatus) and said nothing."
                            : "Exit code \(p.terminationStatus).\n\n" + tail
                    }
                }
                onFinish?(ok ? .finished : .failed)
            }
        }

        do {
            try proc.run()
            process = proc
        } catch {
            isRunning = false
            phase = .failed
            errorText = "Cannot launch Python at \(Self.pythonPath): \(error.localizedDescription)"
            onFinish?(.failed)
        }
    }

    func cancel() {
        guard isRunning else { return }
        wasCancelled = true
        process?.terminate()
        process = nil
        isRunning = false
        phase = .idle
        live.reset(to: .idle)
        errorText = nil
    }

    /// Kill the child before the app goes away.
    ///
    /// A terminated app leaves its subprocess running: whisper carries on with
    /// the whole processor busy and no window left to explain what is using it.
    /// Somebody who quits an app has already said what they want.
    func terminateChild() {
        process?.terminate()
        process = nil
    }

    private func ingest(_ chunk: String) {
        // A pipe hands over bytes, not lines. Splitting the chunk as if it were
        // lines cut one in half wherever the boundary fell, and the half that
        // carried the phase prefix was lost: a stage that takes minutes never
        // ticked. Whatever comes after the last newline waits for the next chunk.
        let combined = residue + chunk
        let lastBreak = combined.lastIndex(of: "\n")
        let text: String
        if let lastBreak {
            text = String(combined[..<lastBreak])
            residue = String(combined[combined.index(after: lastBreak)...])
        } else {
            residue = combined
            // Nothing complete yet. A single line longer than this is the engine
            // printing a progress bar with no newline, so do not hold it for ever.
            guard residue.count > 4096 else { return }
            text = residue
            residue = ""
        }

        var newestPhase: Phase?
        var newestLine: String?
        var newestProgress: Double?

        for line in text.split(separator: "\n").map(String.init) {
            let clean = line.trimmingCharacters(in: .whitespaces)
            guard !clean.isEmpty else { continue }
            // The libraries underneath (torch, speechbrain, lightning) are noisy and print
            // warnings that have nothing to do with the user. Filtering them here instead
            // of showing them is the difference between a readable log and a wall of text.
            if clean.hasPrefix("INFO:") || clean.hasPrefix("WARNING:")
                || clean.contains("UserWarning") || clean.hasPrefix("warn")
                || clean.contains("Lightning automatically") { continue }
            log.append(clean)
            if let p = Phase.from(logLine: clean) { newestPhase = p }
            if let pct = Engine.percentage(in: clean) {
                newestProgress = pct
            } else {
                newestLine = clean
            }
            // Matched against a marker the engine emits, not against its wording.
            // The previous version looked for a phrase from the human message and
            // stopped working the day that sentence was rephrased. It broke silently,
            // on a path no test covers. Keep this in step with ERR_NO_TOKEN in pipeline.py.
            if clean.contains(Engine.errNoToken) {
                errorText = clean.replacingOccurrences(of: Engine.errNoToken, with: "")
                    .trimmingCharacters(in: .whitespaces)
            }
        }
        if log.count > 400 { log.removeFirst(log.count - 400) }

        // A phase change is worth a redraw immediately. Everything else waits for
        // the next tick, because the interface cannot show more than a few frames
        // a second and the engine can produce hundreds of lines in one.
        let now = Date()
        let phaseChanged = newestPhase != nil && newestPhase != phase
        guard phaseChanged || now.timeIntervalSince(lastPublish) > 0.25 else { return }
        lastPublish = now
        if let p = newestPhase, p != phase {
            phase = p
            live.phase = p
            live.progress = nil
        }
        if let pct = newestProgress { live.progress = pct }
        if let line = newestLine { live.lastLine = line }
    }

    /// The percentage whisper prints while it decodes, if this line carries one.
    ///
    /// A bar that moves is the difference between "this is working" and "this has
    /// hung", and the engine has been printing the number all along.
    static func percentage(in line: String) -> Double? {
        // Anchored to what whisper prints. Matching any percentage anywhere meant
        // the confidence figures in the language line drove the bar: two seconds
        // in it read 100% under "Detecting the language", then fell back to
        // nothing, which reads as a bug rather than as progress.
        guard let range = line.range(of: #"Progress:\s*(\d{1,3}(\.\d+)?)%"#,
                                     options: .regularExpression)
        else { return nil }
        let digits = line[range].drop(while: { !$0.isNumber }).prefix(while: { $0.isNumber || $0 == "." })
        guard let value = Double(digits) else { return nil }
        return min(max(value / 100, 0), 1)
    }

    // MARK: - reading the state

    /// Read a job the app already knows the folder of, from the folder.
    ///
    /// The state is two JSON files sitting on disk. Asking the engine for them
    /// meant starting a Python interpreter and importing numpy to hand back
    /// something already written down: three seconds of it, measured, of which
    /// the reading itself was two tenths of a millisecond. Opening a recording
    /// felt broken because it was.
    ///
    /// This duplicates no logic. What must not be written twice is how a job is
    /// derived, and that still happens in exactly one place. Reading a file the
    /// engine wrote is not deriving anything, and if the shape ever changes the
    /// decoder fails loudly and the subprocess path below is still there.
    /// Which job the interface is currently showing. A read that finishes after
    /// the user has moved on, or after a background run has touched something
    /// else, must not land: it used to replace the panel and empty the name
    /// fields somebody was halfway through typing.
    private(set) var showing: String?

    func load(jobDir: String, source: String) {
        showing = jobDir
        let dir = URL(fileURLWithPath: jobDir)
        let fm = FileManager.default

        guard let stateData = fm.contents(atPath: dir.appendingPathComponent("state.json").path),
              let state = try? JSONDecoder().decode(JobState.self, from: stateData)
        else {
            // No readable state on disk. Falling through to the engine costs
            // three seconds and, for a job folder that never got that far, ends
            // in the same silence. Say so instead.
            isLoadingState = false
            info = nil
            speakers = []
            errorText = "This job has no readable state. It was probably "
                      + "interrupted before it wrote anything. Run it again."
            return
        }

        let turnsURL = dir.appendingPathComponent("turns.json")
        let turns = fm.contents(atPath: turnsURL.path)
            .flatMap { try? JSONDecoder().decode([Turn].self, from: $0) } ?? []

        // The engine's own formats, not everything that ended up in the folder.
        // Opening it in the Finder once put a .DS_Store in the list of produced
        // documents, with a button offering to reveal it.
        let written = Set(["md", "txt", "srt", "vtt", "json"])
        let outDir = dir.appendingPathComponent("output")
        let outputs = ((try? fm.contentsOfDirectory(atPath: outDir.path)) ?? [])
            .filter { !$0.hasPrefix(".") && written.contains(($0 as NSString).pathExtension) }
            .sorted()
            .map { outDir.appendingPathComponent($0).path }

        let info = JobInfo(
            jobDir: jobDir, source: source, turns: turns, outputs: outputs,
            dossier: dir.appendingPathComponent("who-is-who.md").path,
            audio: dir.appendingPathComponent("audio16k.wav").path,
            state: state)

        guard showing == jobDir else { return }
        isLoadingState = false
        self.info = info
        speakers = Self.buildSpeakers(from: info)
    }

    /// Read the job state, off the main thread.
    ///
    /// This used to run the subprocess and wait for it right here, on the main actor.
    /// Starting Python and importing numpy costs about four seconds, so dropping a
    /// file froze the whole window for four seconds with nothing on screen to say
    /// why, and the controls that appear once a file is loaded could not draw until
    /// it was over. Under load, with a transcription already running, it is longer.
    func reload(file: URL) {
        isLoadingState = true
        let python = Self.pythonPath
        let root = Self.enginePath

        Task.detached(priority: .userInitiated) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: python)
            proc.arguments = ["-m", "scriba.cli", "info", file.path]
            proc.currentDirectoryURL = URL(fileURLWithPath: root)
            var env = ProcessInfo.processInfo.environment
            env["PYTHONPATH"] = root
            proc.environment = env
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = FileHandle.nullDevice

            var parsed: JobInfo?
            var failure: String?
            do {
                try proc.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                parsed = try? JSONDecoder().decode(JobInfo.self, from: data)
            } catch {
                failure = "Cannot read the state: \(error.localizedDescription)"
            }

            let result = parsed
            let problem = failure
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.isLoadingState = false
                if let problem { self.errorText = problem }
                guard let result else { return }
                // Same rule as load(jobDir:). A read that comes back after the
                // selection moved on belongs to a different recording.
                guard self.showing == nil || self.showing == result.jobDir else { return }
                self.info = result
                self.speakers = Self.buildSpeakers(from: result)
            }
        }
    }

    /// Builds the speaker table from the turns. For each speaker it keeps the longest
    /// turn aside: that is the one to replay when deciding whose voice it is.
    static func buildSpeakers(from info: JobInfo) -> [Speaker] {
        var seconds: [String: Double] = [:]
        var counts: [String: Int] = [:]
        var best: [String: Turn] = [:]

        for t in info.turns {
            guard let s = t.speaker else { continue }
            seconds[s, default: 0] += t.end - t.start
            counts[s, default: 0] += 1
            if let cur = best[s] {
                if t.text.count > cur.text.count { best[s] = t }
            } else {
                best[s] = t
            }
        }

        let names = info.state.names ?? [:]
        let matches = info.state.matches ?? [:]

        return seconds.keys.sorted().map { id in
            let quote = best[id]
            let match = matches[id]
            return Speaker(
                id: id,
                // Only a certain match fills the field. A borderline candidate stays a
                // suggestion to accept by hand: pre-filling it would get the uncertain
                // cases accepted out of inertia, which is exactly the wrong outcome.
                name: names[id] ?? match?.name ?? "",
                suggested: match?.name ?? match?.candidate,
                score: match?.score ?? 0,
                reason: match?.reason ?? "",
                speechSeconds: seconds[id] ?? 0,
                turnCount: counts[id] ?? 0,
                previewStart: quote?.start ?? 0,
                previewEnd: min((quote?.start ?? 0) + 12, quote?.end ?? 12),
                longestQuote: quote?.text ?? ""
            )
        }
        .sorted { $0.speechSeconds > $1.speechSeconds }
    }
}
