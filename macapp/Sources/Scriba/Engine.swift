import Foundation

/// Bridge to the Python engine.
///
/// The app reimplements nothing: `scriba` does the work, this class launches it,
/// reads its output to follow where it has got to, and re-reads the state once it is done.
/// That is deliberate. The engine has to stay usable on its own from the terminal,
/// and an app that duplicates the logic starts drifting the day after.
@MainActor
final class Engine: ObservableObject {

    @Published var phase: Phase = .idle
    @Published var log: [String] = []
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

    func applyNames(file: URL, mapping: [String: String], enroll: Bool) {
        guard !isRunning else { return }
        var args = ["-m", "scriba.cli", "name", file.path]
        args += mapping.filter { !$0.value.isEmpty }.map { "\($0.key)=\($0.value)" }
        if !enroll { args.append("--no-enroll") }
        launch(args, on: file, startingPhase: .identifying)
    }

    private func launch(_ args: [String], on file: URL, startingPhase: Phase,
                        onFinish: ((Outcome) -> Void)? = nil) {
        isRunning = true
        wasCancelled = false
        errorText = nil
        log.removeAll()
        phase = startingPhase

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
            Task { @MainActor in
                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                self?.isRunning = false
                self?.process = nil
                if self?.wasCancelled == true {
                    self?.wasCancelled = false
                    self?.phase = .idle
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
                    if self?.errorText == nil {
                        self?.errorText = self?.log.suffix(6).joined(separator: "\n")
                            ?? "The engine exited with code \(p.terminationStatus)."
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

    private func ingest(_ text: String) {
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
            if let p = Phase.from(logLine: clean) { phase = p }
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
    }

    // MARK: - reading the state

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
