import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var engine = Engine()
    @State private var file: URL?
    @State private var language = "auto"
    @State private var expectedSpeakers = 0          // 0 = let pyannote decide
    @State private var enrollOnSave = true
    @State private var showLog = false

    private let languages = [("auto", "Detect automatically"), ("it", "Italian"),
                             ("es", "Spanish"), ("en", "English"), ("fr", "French"),
                             ("de", "German"), ("pt", "Portuguese")]

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 260, ideal: 290, max: 340)
        } detail: {
            detail
        }
        .frame(minWidth: 940, minHeight: 620)
        .alert("Something went wrong",
               isPresented: .constant(engine.errorText != nil),
               presenting: engine.errorText) { _ in
            Button("Close") { engine.errorText = nil }
        } message: { text in
            Text(text)
        }
    }

    // MARK: - left column

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 18) {
            DropZone(file: $file) { picked in
                file = picked
                engine.reload(file: picked)
            }

            if file != nil {
                VStack(alignment: .leading, spacing: 12) {
                    Picker("Language", selection: $language) {
                        ForEach(languages, id: \.0) { Text($0.1).tag($0.0) }
                    }
                    Text("Pick the wrong language and Whisper does not fail: it translates by ear and invents. When in doubt, let it detect the language on its own.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    Stepper(expectedSpeakers == 0
                            ? "People: as many as it finds"
                            : "People: \(expectedSpeakers)",
                            value: $expectedSpeakers, in: 0...12)
                    Text("If you know how many there are, saying so improves voice separation a lot.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.horizontal, 4)

                Button {
                    guard let f = file else { return }
                    engine.run(file: f, language: language,
                               minSpeakers: expectedSpeakers > 0 ? expectedSpeakers : nil,
                               maxSpeakers: expectedSpeakers > 0 ? expectedSpeakers : nil)
                } label: {
                    Label(engine.info == nil ? "Transcribe" : "Reprocess",
                          systemImage: "waveform.badge.magnifyingglass")
                        .frame(maxWidth: .infinity)
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                .disabled(engine.isRunning)

                if engine.isRunning {
                    Button("Stop", role: .destructive) { engine.cancel() }
                        .frame(maxWidth: .infinity)
                }
            }

            Spacer()

            if !engine.log.isEmpty {
                DisclosureGroup("Technical details", isExpanded: $showLog) {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 2) {
                            ForEach(Array(engine.log.suffix(60).enumerated()), id: \.offset) { _, l in
                                Text(l).font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(height: 130)
                }
                .font(.caption)
            }
        }
        .padding(18)
    }

    // MARK: - right column

    @ViewBuilder
    private var detail: some View {
        if engine.isRunning {
            ProgressPanel(phase: engine.phase, log: engine.log)
        } else if let info = engine.info, !engine.speakers.isEmpty {
            ResultPanel(engine: engine, info: info, file: file,
                        enrollOnSave: $enrollOnSave)
        } else if file != nil {
            Placeholder(
                icon: "waveform",
                title: "Ready",
                message: "Press Transcribe. The first run on a ten-minute file takes a few minutes: the large model runs on the CPU."
            )
        } else {
            Placeholder(
                icon: "square.and.arrow.down",
                title: "Drop a recording here",
                message: "Voice memos, .m4a, .mp3, .wav, or a video. The file never leaves the Mac: transcription and voice identification run locally."
            )
        }
    }
}

// MARK: - drop zone

struct DropZone: View {
    @Binding var file: URL?
    var onPick: (URL) -> Void
    @State private var isTargeted = false

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: file == nil ? "tray.and.arrow.down" : "waveform.circle.fill")
                .font(.system(size: 30))
                .foregroundStyle(isTargeted ? Color.accentColor : .secondary)
            Text(file?.lastPathComponent ?? "Drop an audio file")
                .font(.callout)
                .multilineTextAlignment(.center)
                .lineLimit(3)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 26)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [6]))
                .foregroundStyle(isTargeted ? Color.accentColor : Color.secondary.opacity(0.4))
        )
        .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
            guard let p = providers.first else { return false }
            _ = p.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                DispatchQueue.main.async { onPick(url) }
            }
            return true
        }
        .onTapGesture {
            let panel = NSOpenPanel()
            panel.allowedContentTypes = [.audio, .movie, .mpeg4Audio, .mp3, .wav]
            panel.allowsMultipleSelection = false
            if panel.runModal() == .OK, let url = panel.url { onPick(url) }
        }
    }
}

// MARK: - progress

struct ProgressPanel: View {
    let phase: Phase
    let log: [String]

    private let order: [Phase] = [.preparing, .detecting, .transcribing,
                                  .aligning, .diarizing, .identifying, .exporting]

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text(phase.rawValue).font(.title2).bold()
            ProgressView().progressViewStyle(.linear)

            VStack(alignment: .leading, spacing: 10) {
                ForEach(order, id: \.self) { p in
                    let idx = order.firstIndex(of: p) ?? 0
                    let cur = order.firstIndex(of: phase) ?? 0
                    HStack(spacing: 10) {
                        Image(systemName: idx < cur ? "checkmark.circle.fill"
                              : idx == cur ? "circle.dotted" : "circle")
                            .foregroundStyle(idx < cur ? .green : idx == cur ? .primary : .secondary)
                        Text(p.rawValue)
                            .foregroundStyle(idx <= cur ? .primary : .secondary)
                    }
                }
            }
            if let last = log.last {
                Text(last).font(.caption).foregroundStyle(.secondary)
                    .lineLimit(2).textSelection(.enabled)
            }
            Spacer()
        }
        .padding(30)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct Placeholder: View {
    let icon: String, title: String, message: String
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: icon).font(.system(size: 44)).foregroundStyle(.secondary)
            Text(title).font(.title3).bold()
            Text(message)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}
