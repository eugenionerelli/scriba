import SwiftUI

@main
struct ScribaApp: App {
    var body: some Scene {
        WindowGroup("Scriba") {
            ContentView()
        }
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) { }
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

struct SettingsView: View {
    @State private var python = Engine.pythonPath
    @State private var engine = Engine.enginePath

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
