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
        case "transcribed": return "Transcribed, no output yet"
        case "voices only": return "Voices found, not transcribed"
        case "text only":   return "Text only"
        default:            return "Not started"
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

    func reload() {
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

            var loaded: [JobSummary] = []
            if (try? proc.run()) != nil {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                loaded = (try? JSONDecoder().decode([JobSummary].self, from: data)) ?? []
            }

            let result = loaded
            await MainActor.run { [weak self] in
                self?.jobs = result
                self?.isLoading = false
            }
        }
    }
}
