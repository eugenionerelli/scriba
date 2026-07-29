import SwiftUI

@main
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
    /// Posted by the Refresh command. The window listens; the menu item does not
    /// need to know anything about it.
    static let scribaRefresh = Notification.Name("dev.nerelli.scriba.refresh")
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
            Section("pyannote token") {
                Text("The Hugging Face token is set once from the terminal:")
                    .font(.callout)
                Text("scriba token hf_xxxxxxxx")
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                Text("It ends up in the Keychain, not in a file.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 520, height: 320)
    }
}
