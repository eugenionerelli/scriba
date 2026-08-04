import SwiftUI

/// The entry point picks between the window and the one command-line check.
///
/// SwiftUI's App protocol supplies its own `main`, so intercepting the arguments
/// means owning the entry point and calling it. The check has to live in this
/// binary rather than a tool of its own: it drives LiveTranscriber, and a second
/// executable target cannot depend on an executable target.
@main
struct Entry {
    static func main() {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--live-check"), i + 1 < args.count {
            var language = "en"
            if let l = args.firstIndex(of: "--lang"), l + 1 < args.count {
                language = args[l + 1]
            }
            LiveCheck.run(path: args[i + 1], language: language)
        }
        ScribaApp.main()
    }
}

struct ScribaApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup("Scriba") {
            ContentView()
        }
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) { }
            CommandGroup(after: .toolbar) {
                Button("Refresh the list") {
                    NotificationCenter.default.post(name: .scribaRefresh, object: nil)
                }
                .keyboardShortcut("r", modifiers: .command)
            }
            // The two things this app does, on the keys a Mac application puts
            // them on. Reaching for the mouse to start something that then runs
            // for an hour is the wrong shape.
            CommandMenu("Transcribe") {
                Button("Record a conversation") {
                    NotificationCenter.default.post(name: .scribaRecord, object: nil)
                }
                // Shift is not decoration: command-R is Refresh, and starting a
                // recording by accident while reaching for it would be worse than
                // the other way round.
                .keyboardShortcut("r", modifiers: [.command, .shift])
                Divider()
                Button("Start the queue") {
                    NotificationCenter.default.post(name: .scribaStart, object: nil)
                }
                .keyboardShortcut("t", modifiers: .command)
                Button("Stop") {
                    NotificationCenter.default.post(name: .scribaStop, object: nil)
                }
                .keyboardShortcut(".", modifiers: .command)
            }
            if WindowShot.isEnabled {
                CommandGroup(after: .saveItem) {
                    Button("Save window as PNG") { WindowShot.save() }
                        .keyboardShortcut("p", modifiers: [.command, .option])
                }
            }
        }

        Settings {
            SettingsView()
        }
    }
}

extension Notification.Name {
    /// Posted by the menu commands. The window listens; the menu items do not need
    /// to know anything about the queue.
    static let scribaRefresh = Notification.Name("dev.nerelli.scriba.refresh")
    static let scribaStart = Notification.Name("dev.nerelli.scriba.start")
    static let scribaStop = Notification.Name("dev.nerelli.scriba.stop")
    static let scribaRecord = Notification.Name("dev.nerelli.scriba.record")
}

/// Exists for one line: quitting has to take the engine with it.
final class AppDelegate: NSObject, NSApplicationDelegate {
    static var onQuit: (() -> Void)?

    func applicationWillTerminate(_ notification: Notification) {
        AppDelegate.onQuit?()
    }
}

struct SettingsView: View {
    @State private var python = Engine.pythonPath
    @State private var engine = Engine.enginePath
    @AppStorage("liveTextEnabled") private var liveText = false
    @State private var token = ""
    @State private var tokenState: TokenState = .unknown

    enum TokenState: Equatable {
        case unknown, present, saved, missing, failed(String)
    }

    private var engineFolderState: String {
        var isDir: ObjCBool = false
        let there = FileManager.default.fileExists(atPath: engine, isDirectory: &isDir)
        if !there || !isDir.boolValue {
            return "No folder there. Leave it as it is if you installed scriba with "
                 + "pip: this only matters when you are running it from a clone."
        }
        return FileManager.default.fileExists(atPath: engine + "/scriba/cli.py")
            ? "Found." : "Folder found, but no scriba package inside it."
    }

    var body: some View {
        Form {
            Section("Engine") {
                TextField("Python interpreter", text: $python)
                    .onChange(of: python) { _, v in Engine.pythonPath = v }
                Text(FileManager.default.isExecutableFile(atPath: python)
                     ? "Found."
                     : "Not found. It has to be the python from the conda environment where whisperx is installed.")
                    .font(.caption)
                    .foregroundStyle(FileManager.default.isExecutableFile(atPath: python)
                                     ? Color.secondary : Color.red)

                TextField("scriba package folder", text: $engine)
                    .onChange(of: engine) { _, v in Engine.enginePath = v }
                // This one is the process working directory, so a wrong value
                // fails before Python is reached and the error blames the
                // interpreter. It used to be the field without validation, which
                // is the wrong way round.
                Text(engineFolderState)
                    .font(.caption)
                    .foregroundStyle(Engine.isEngineFolderUsable ? Color.secondary : Color.red)
            }
            Section("Recording") {
                Toggle("Show the words while recording", isOn: $liveText)
                Text("A preview from the speech model built into macOS 26, about a "
                     + "second behind the speaker and with its own punctuation. It is "
                     + "not kept: the document is still produced afterwards by "
                     + "whisper, which makes fewer mistakes. Set this before you "
                     + "start; it cannot be changed mid-recording.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("pyannote token") {
                // A secure field, and the value never leaves this window: it goes
                // straight into the Keychain through the Security framework. The
                // old instruction here was to type `scriba token hf_...` into a
                // terminal, which leaves the secret in the shell history.
                SecureField("hf_…", text: $token)
                    .onSubmit(saveToken)
                HStack {
                    Button("Save in the Keychain", action: saveToken)
                        .disabled(token.trimmingCharacters(in: .whitespaces).isEmpty)
                    if tokenState == .present || tokenState == .saved {
                        Button("Remove") {
                            Keychain.forget()
                            token = ""
                            tokenState = .missing
                        }
                    }
                    Spacer()
                }
                Text(tokenMessage)
                    .font(.caption)
                    .foregroundStyle(tokenColour)
            }
        }
        .formStyle(.grouped)
        .frame(width: 520, height: 480)
        .onAppear { tokenState = Keychain.hasToken() ? .present : .missing }
    }

    private func saveToken() {
        if let problem = Keychain.save(token) {
            tokenState = .failed(problem)
        } else {
            // Clear the field on success. Leaving the token sitting in a window
            // that anybody walking past can reveal is the thing this section
            // exists to avoid.
            token = ""
            tokenState = .saved
        }
    }

    private var tokenMessage: String {
        switch tokenState {
        case .unknown: return ""
        case .present: return "A token is stored. Type a new one to replace it."
        case .saved: return "Saved. Diarization will work on the next run."
        case .missing:
            return "No token yet. Without one the recordings are transcribed but not "
                 + "separated by speaker. Get one from huggingface.co/settings/tokens "
                 + "and accept the conditions on pyannote/speaker-diarization-community-1."
        case .failed(let why): return why
        }
    }

    private var tokenColour: Color {
        switch tokenState {
        case .failed: return .red
        case .saved, .present: return .secondary
        default: return .secondary
        }
    }
}
