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

    private var process: Process?

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

    /// Mirrors ERR_NO_TOKEN in scriba/pipeline.py. The one string that genuinely
    /// crosses the Python/Swift boundary, so it is spelled out in one place on
    /// each side rather than buried inside a condition.
    static let errNoToken = "[scriba:error:hf-token]"

    // MARK: - execution

    func run(file: URL, language: String, minSpeakers: Int?, maxSpeakers: Int?) {
        guard !isRunning else { return }
        var args = ["-m", "scriba.cli", "run", file.path, "--lang", language]
        if let m = minSpeakers { args += ["--min-speakers", String(m)] }
        if let m = maxSpeakers { args += ["--max-speakers", String(m)] }
        launch(args, on: file, startingPhase: .preparing)
    }

    func applyNames(file: URL, mapping: [String: String], enroll: Bool) {
        guard !isRunning else { return }
        var args = ["-m", "scriba.cli", "name", file.path]
        args += mapping.filter { !$0.value.isEmpty }.map { "\($0.key)=\($0.value)" }
        if !enroll { args.append("--no-enroll") }
        launch(args, on: file, startingPhase: .identifying)
    }

    private func launch(_ args: [String], on file: URL, startingPhase: Phase) {
        isRunning = true
        errorText = nil
        log.removeAll()
        phase = startingPhase

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: Self.pythonPath)
        proc.arguments = args
        proc.currentDirectoryURL = URL(fileURLWithPath: Self.enginePath)

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
                if p.terminationStatus == 0 {
                    self?.phase = .done
                    self?.reload(file: file)
                } else {
                    self?.phase = .failed
                    if self?.errorText == nil {
                        self?.errorText = self?.log.suffix(6).joined(separator: "\n")
                            ?? "The engine exited with code \(p.terminationStatus)."
                    }
                }
            }
        }

        do {
            try proc.run()
            process = proc
        } catch {
            isRunning = false
            phase = .failed
            errorText = "Cannot launch Python at \(Self.pythonPath): \(error.localizedDescription)"
        }
    }

    func cancel() {
        process?.terminate()
        process = nil
        isRunning = false
        phase = .idle
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

    func reload(file: URL) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: Self.pythonPath)
        proc.arguments = ["-m", "scriba.cli", "info", file.path]
        proc.currentDirectoryURL = URL(fileURLWithPath: Self.enginePath)
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = Self.enginePath
        proc.environment = env
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            proc.waitUntilExit()
            guard let parsed = try? JSONDecoder().decode(JobInfo.self, from: data) else { return }
            info = parsed
            speakers = Self.buildSpeakers(from: parsed)
        } catch {
            errorText = "Cannot read the state: \(error.localizedDescription)"
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
