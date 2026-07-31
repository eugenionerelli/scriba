import Foundation

/// One recording scriba knows about, whether it has been processed or not.
struct JobSummary: Codable, Identifiable, Hashable {
    var id: String { jobDir }
    let jobDir: String
    let source: String
    let sourcePath: String
    let recorded: String
    let duration: Double
    let state: String
    let names: [String: String]
    let speakers: Int
    let sizeMb: Double
    let hasOutput: Bool

    enum CodingKeys: String, CodingKey {
        case jobDir = "job_dir"
        case source
        case sourcePath = "source_path"
        case recorded, duration, state, names, speakers
        case sizeMb = "size_mb"
        case hasOutput = "has_output"
    }

    /// What to show in the list. The engine's words are for a terminal.
    var label: String {
        switch state {
        case "done":        return "Ready to read"
        case "transcribed": return "Transcribed, no document written"
        case "voices only": return "Voices separated, words missing"
        case "text only":   return "Words only, voices not separated"
        default:            return "Started and produced nothing"
        }
    }

    var isFinished: Bool { state == "done" }
}

/// A recording waiting to be processed, or being processed now.
struct QueueItem: Identifiable, Hashable {
    enum State: Hashable {
        case waiting
        case running
        case finished
        case failed(String)
    }
    let id = UUID()
    let url: URL
    var state: State = .waiting
    /// Read once when the file joins the queue. Reading it from the estimate
    /// meant opening every queued file's container on the main thread on every
    /// redraw, which is why dropping a batch of recordings froze the window.
    var minutes: Double?
}

/// The list of everything scriba has touched.
///
/// Read through `scriba jobs list --json` rather than by poking at the job folder
/// from Swift. One source of truth for what a job is, and the CLI stays the thing
/// that defines it.
@MainActor
final class JobsStore: ObservableObject {
    @Published var jobs: [JobSummary] = []
    @Published var isLoading = false
    /// Set when the engine could not be asked. The difference between "you have
    /// no recordings" and "I could not find out" matters: the second one used to
    /// be shown as the first, on a machine holding dozens of transcripts.
    @Published var problem: String?

    /// For callers that need the list to be current before they act on it.
    func reloadAndWait() async {
        reload()
        while isLoading { try? await Task.sleep(nanoseconds: 40_000_000) }
    }

    func reload() {
        // Reading the list starts a Python interpreter, which takes about three
        // seconds. Command-tabbing in and out a few times used to put three or
        // four of them on the processor at once, each one racing to publish.
        guard !isLoading else { return }
        isLoading = true
        let python = Engine.pythonPath
        let root = Engine.enginePath

        Task.detached(priority: .utility) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: python)
            proc.arguments = ["-m", "scriba.cli", "jobs", "list", "--json"]
            proc.currentDirectoryURL = URL(fileURLWithPath: root)
            var env = ProcessInfo.processInfo.environment
            env["PYTHONPATH"] = root
            proc.environment = env
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = FileHandle.nullDevice

            var loaded: [JobSummary]?
            var failure: String?
            if (try? proc.run()) != nil {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                if proc.terminationStatus != 0 {
                    failure = "The engine exited with code \(proc.terminationStatus) "
                            + "when asked for the list of recordings."
                } else {
                    loaded = try? JSONDecoder().decode([JobSummary].self, from: data)
                    if loaded == nil { failure = "The list of recordings could not be read." }
                }
            } else {
                failure = "Could not run \(python). Open Settings and check the path."
            }

            let result = loaded
            let problem = failure
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.isLoading = false
                self.problem = problem
                // Only replace the list with one that was actually read. A failed
                // read used to empty the sidebar, and the app looked like a fresh
                // install with nothing in it.
                if let result, result != self.jobs { self.jobs = result }
            }
        }
    }
}
