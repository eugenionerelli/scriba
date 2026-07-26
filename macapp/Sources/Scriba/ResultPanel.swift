import SwiftUI
import AVFoundation
import AppKit

/// The panel where you decide who is who.
///
/// The rule this panel follows: nobody writes a name without having heard the
/// voice first. That is why every row keeps the listen button right next to the
/// name field. If assigning a name costs less than checking it, people assign
/// without checking, and the wrong transcript ends up in NotebookLM as if it
/// were a fact.
struct ResultPanel: View {
    @ObservedObject var engine: Engine
    let info: JobInfo
    let file: URL?
    @Binding var enrollOnSave: Bool

    @State private var drafts: [String: String] = [:]
    @State private var player: AVAudioPlayer?
    @State private var playing: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                header
                Divider()
                speakerSection
                Divider()
                outputSection
                if !info.turns.isEmpty {
                    Divider()
                    previewSection
                }
            }
            .padding(28)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onAppear(perform: seedDrafts)
        .onChange(of: engine.speakers) { _, _ in seedDrafts() }
    }

    // MARK: - header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(URL(fileURLWithPath: info.source).lastPathComponent)
                .font(.title2).bold()
            HStack(spacing: 14) {
                Label(humanDuration(info.state.duration ?? 0), systemImage: "clock")
                Label(languageLabel, systemImage: "globe")
                Label("\(engine.speakers.count) speakers", systemImage: "person.2")
            }
            .font(.callout)
            .foregroundStyle(.secondary)

            if let note = info.state.languageNote, let conf = info.state.languageConfidence,
               conf < 0.6 {
                Label(note, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    private var languageLabel: String {
        let map = ["it": "Italian", "es": "Spanish", "en": "English",
                   "fr": "French", "de": "German", "pt": "Portuguese"]
        let code = info.state.language ?? "?"
        return map[code] ?? code
    }

    // MARK: - speakers

    private var speakerSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Who is speaking").font(.headline)

            ForEach(engine.speakers) { s in
                SpeakerRow(
                    speaker: s,
                    name: Binding(
                        get: { drafts[s.id] ?? s.name },
                        set: { drafts[s.id] = $0 }
                    ),
                    isPlaying: playing == s.id,
                    onPlay: { play(s) }
                )
            }

            Toggle("Learn these voices for the next recordings", isOn: $enrollOnSave)
                .font(.callout)
            Text("Saves a voice print tied to the name. The next time this person shows up in a recording, the name attaches on its own.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Button {
                guard let f = file else { return }
                engine.applyNames(file: f, mapping: drafts, enroll: enrollOnSave)
            } label: {
                Label("Save the names and rewrite the files", systemImage: "checkmark.circle")
            }
            .controlSize(.large)
            .buttonStyle(.borderedProminent)
            .disabled(engine.isRunning || drafts.values.allSatisfy(\.isEmpty))
        }
    }

    // MARK: - produced files

    private var outputSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("To upload to NotebookLM").font(.headline)

            if let nb = info.outputs.first(where: { $0.contains("NotebookLM") }) {
                HStack(spacing: 10) {
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: nb)])
                    } label: {
                        Label("Show in Finder", systemImage: "folder")
                    }
                    Button {
                        let text = (try? String(contentsOfFile: nb, encoding: .utf8)) ?? ""
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(text, forType: .string)
                    } label: {
                        Label("Copy the text", systemImage: "doc.on.clipboard")
                    }
                }
                Text(URL(fileURLWithPath: nb).lastPathComponent)
                    .font(.caption).foregroundStyle(.secondary)
            }

            DisclosureGroup("Other formats (\(max(info.outputs.count - 1, 0)))") {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(info.outputs.filter { !$0.contains("NotebookLM") }, id: \.self) { p in
                        Button(URL(fileURLWithPath: p).lastPathComponent) {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: p)])
                        }
                        .buttonStyle(.link)
                        .font(.caption)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
            }
            .font(.callout)
        }
    }

    // MARK: - preview

    private var previewSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Preview").font(.headline)
            ForEach(info.turns.prefix(12)) { t in
                let who = drafts[t.speaker ?? ""].flatMap { $0.isEmpty ? nil : $0 }
                    ?? (t.speaker ?? "?").replacingOccurrences(of: "SPEAKER_", with: "Voice ")
                (Text("\(who) ").bold().foregroundColor(.accentColor)
                 + Text("[\(timecode(t.start))] ").font(.caption).foregroundColor(.secondary)
                 + Text(t.text))
                    .font(.callout)
                    .textSelection(.enabled)
            }
            if info.turns.count > 12 {
                Text("…and \(info.turns.count - 12) more turns")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - actions

    private func seedDrafts() {
        for s in engine.speakers where drafts[s.id] == nil {
            drafts[s.id] = s.name
        }
    }

    /// Plays the longest stretch of this voice. The job's 16 kHz WAV is already
    /// there, so there is no need to decode the original again.
    private func play(_ s: Speaker) {
        player?.stop()
        if playing == s.id { playing = nil; return }
        guard let p = try? AVAudioPlayer(contentsOf: URL(fileURLWithPath: info.audio)) else { return }
        player = p
        p.currentTime = s.previewStart
        p.play()
        playing = s.id
        let window = max(s.previewEnd - s.previewStart, 3)
        DispatchQueue.main.asyncAfter(deadline: .now() + window) {
            if playing == s.id { p.stop(); playing = nil }
        }
    }
}

// MARK: - single row

struct SpeakerRow: View {
    let speaker: Speaker
    @Binding var name: String
    let isPlaying: Bool
    let onPlay: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Button(action: onPlay) {
                    Image(systemName: isPlaying ? "stop.circle.fill" : "play.circle.fill")
                        .font(.title2)
                }
                .buttonStyle(.plain)
                .help("Listen to the longest stretch of speech from this voice")

                VStack(alignment: .leading, spacing: 2) {
                    Text(speaker.id.replacingOccurrences(of: "SPEAKER_", with: "Voice "))
                        .font(.callout).bold()
                    Text("\(humanDuration(speaker.speechSeconds)) · \(speaker.turnCount) turns")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .frame(width: 150, alignment: .leading)

                TextField("Name", text: $name)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 220)

                if speaker.isConfirmedMatch {
                    Label("from the voice registry · \(String(format: "%.2f", speaker.score))",
                          systemImage: "sparkles")
                        .font(.caption)
                        .foregroundStyle(.purple)
                        .help(speaker.reason)
                } else if let proposta = speaker.pendingSuggestion {
                    // A suggestion, not a decision: the name lands in the field only
                    // if you click it, after you have listened.
                    Button("Maybe “\(proposta)”?") { name = proposta }
                        .buttonStyle(.link)
                        .font(.caption)
                        .help(speaker.reason)
                }
            }

            if !speaker.longestQuote.isEmpty {
                Text("“\(speaker.longestQuote.prefix(180))\(speaker.longestQuote.count > 180 ? "…" : "")”")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .padding(.leading, 42)
            }
            // When diarization produces a voice with very little speech it is almost
            // always an artifact. The sounds are real, somebody in the room murmured
            // them; the extra speaker label is what has no referent. Saying so here
            // keeps that label from getting a name.
            if speaker.speechSeconds < 25 {
                Label("Very little speech. This may be backchannel sounds from someone else rather than a person of its own. Listen again before giving it a name.",
                      systemImage: "questionmark.circle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .padding(.leading, 42)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.secondary.opacity(0.06)))
    }
}
