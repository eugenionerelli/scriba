import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

/// The window.
///
/// The sidebar holds two lists: recordings waiting to be processed, and everything
/// processed before. The detail pane shows whatever is selected.
///
/// Nothing starts on its own. Dropping a file adds it to a queue and stops there,
/// because transcription runs for as long as the recording lasts and is not something
/// anybody should begin by accident.
struct ContentView: View {
    @StateObject private var engine = Engine()
    @StateObject private var store = JobsStore()

    @State private var queue: [QueueItem] = []
    @State private var selection: Selection?
    @State private var language = "auto"
    @State private var expectedSpeakers = 0
    @State private var enrollOnSave = true
    @State private var isTargeted = false
    @State private var runningAll = false

    enum Selection: Hashable {
        case pending(UUID)
        case job(String)
    }

    private let languages = [("auto", "Detect automatically"), ("it", "Italian"),
                             ("es", "Spanish"), ("en", "English"), ("fr", "French"),
                             ("de", "German"), ("pt", "Portuguese")]

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 260, ideal: 300, max: 380)
        } detail: {
            detail
        }
        .frame(minWidth: 900, minHeight: 560)
        .onAppear { store.reload() }
        .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
            accept(providers)
        }
        .overlay {
            if isTargeted {
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(Color.accentColor, lineWidth: 3)
                    .padding(6)
                    .allowsHitTesting(false)
            }
        }
        .alert("Something went wrong",
               isPresented: .constant(engine.errorText != nil),
               presenting: engine.errorText) { _ in
            Button("Close") { engine.errorText = nil }
        } message: { Text($0) }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    let panel = NSOpenPanel()
                    panel.allowedContentTypes = [.audio, .movie]
                    panel.allowsMultipleSelection = true
                    if panel.runModal() == .OK { add(panel.urls) }
                } label: {
                    Label("Add recordings", systemImage: "plus")
                }
            }
        }
    }

    // MARK: - sidebar

    private var sidebar: some View {
        List(selection: $selection) {
            Section("Waiting") {
                if queue.isEmpty {
                    Text("Drop recordings anywhere in this window")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(queue) { item in
                        QueueRow(item: item).tag(Selection.pending(item.id))
                    }
                    .onDelete { queue.remove(atOffsets: $0) }
                }
            }

            Section("Processed") {
                if store.jobs.isEmpty && !store.isLoading {
                    Text("Nothing yet").font(.callout).foregroundStyle(.secondary)
                }
                ForEach(store.jobs.filter { $0.duration > 0 }) { job in
                    JobRowView(job: job).tag(Selection.job(job.jobDir))
                }
            }
        }
        .safeAreaInset(edge: .bottom) {
            if !queue.isEmpty {
                VStack(spacing: 8) {
                    Divider()
                    Button {
                        startAll()
                    } label: {
                        Label(queue.count == 1
                              ? "Transcribe this one"
                              : "Transcribe all \(queue.count)",
                              systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)
                    .disabled(engine.isRunning)

                    if !estimate.isEmpty {
                        Text(estimate)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                }
                .padding(12)
                .background(.bar)
            }
        }
    }

    /// Say how long it will take before it starts, not after. Transcription runs for
    /// roughly the length of the recording, and somebody who does not know that reads
    /// a still progress bar as a hang.
    private var estimate: String {
        let waiting = queue.filter { $0.state == .waiting }
        guard !waiting.isEmpty else { return "" }
        let minutes = waiting.compactMap { durationMinutes(of: $0.url) }.reduce(0, +)
        guard minutes > 0 else { return "Runs for about as long as the recordings last." }
        return "About \(Int(minutes.rounded())) minutes of work, roughly the length of the audio."
    }

    private func durationMinutes(of url: URL) -> Double? {
        let asset = AVURLAsset(url: url)
        let seconds = CMTimeGetSeconds(asset.duration)
        return seconds.isFinite && seconds > 0 ? seconds / 60 : nil
    }

    // MARK: - detail

    @ViewBuilder
    private var detail: some View {
        if engine.isRunning {
            ProgressPanel(phase: engine.phase, log: engine.log,
                          current: currentlyRunning,
                          remaining: queue.filter { $0.state == .waiting }.count)
        } else if case .pending(let id) = selection,
                  let item = queue.first(where: { $0.id == id }) {
            PendingPanel(item: item, language: $language,
                         expectedSpeakers: $expectedSpeakers,
                         languages: languages, onStart: { start(item) })
        } else if case .job(let dir) = selection,
                  let job = store.jobs.first(where: { $0.jobDir == dir }) {
            if let info = engine.info, info.jobDir == dir, !engine.speakers.isEmpty {
                ResultPanel(engine: engine, info: info,
                            file: URL(fileURLWithPath: job.sourcePath),
                            enrollOnSave: $enrollOnSave)
            } else {
                JobPanel(job: job, loading: engine.isLoadingState)
                    .onAppear {
                        engine.reload(file: URL(fileURLWithPath: job.sourcePath))
                    }
            }
        } else {
            Welcome(processed: store.jobs.count)
        }
    }

    private var currentlyRunning: String {
        queue.first(where: { $0.state == .running })?.url.lastPathComponent ?? ""
    }

    // MARK: - actions

    private func accept(_ providers: [NSItemProvider]) -> Bool {
        var found = false
        for provider in providers where provider.canLoadObject(ofClass: URL.self) {
            found = true
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                DispatchQueue.main.async { add([url]) }
            }
        }
        return found
    }

    private func add(_ urls: [URL]) {
        let known = Set(queue.map(\.url))
        for url in urls where !known.contains(url) {
            queue.append(QueueItem(url: url))
        }
        if selection == nil, let first = queue.first {
            selection = .pending(first.id)
        }
    }

    private func start(_ item: QueueItem) {
        guard !engine.isRunning else { return }
        mark(item.id, .running)
        engine.run(file: item.url, language: language,
                   minSpeakers: expectedSpeakers > 0 ? expectedSpeakers : nil,
                   maxSpeakers: expectedSpeakers > 0 ? expectedSpeakers : nil) { ok in
            mark(item.id, ok ? .finished : .failed("did not finish"))
            store.reload()
            if runningAll { next() }
        }
    }

    private func startAll() {
        runningAll = queue.filter { $0.state == .waiting }.count > 1
        next()
    }

    /// One at a time. Whisper already spreads across every fast CPU, so two
    /// transcriptions at once finish later than the same two in sequence, and the
    /// machine is unusable while they run.
    private func next() {
        guard let item = queue.first(where: { $0.state == .waiting }) else {
            runningAll = false
            return
        }
        start(item)
    }

    private func mark(_ id: UUID, _ state: QueueItem.State) {
        guard let i = queue.firstIndex(where: { $0.id == id }) else { return }
        queue[i].state = state
    }
}

// MARK: - rows

struct QueueRow: View {
    let item: QueueItem
    var body: some View {
        HStack(spacing: 8) {
            switch item.state {
            case .waiting:  Image(systemName: "clock").foregroundStyle(.secondary)
            case .running:  ProgressView().controlSize(.small)
            case .finished: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            case .failed:   Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(item.url.lastPathComponent).lineLimit(1)
                if case .failed(let why) = item.state {
                    Text(why).font(.caption).foregroundStyle(.orange)
                }
            }
        }
    }
}

struct JobRowView: View {
    let job: JobSummary
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: job.isFinished ? "doc.text.fill" : "waveform")
                .foregroundStyle(job.isFinished ? Color.accentColor : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(job.source).lineLimit(1)
                HStack(spacing: 6) {
                    if !job.recorded.isEmpty { Text(job.recorded) }
                    if job.duration > 0 { Text("\(Int(job.duration / 60)) min") }
                    if !job.names.isEmpty {
                        Text(job.names.values.sorted().joined(separator: ", ")).lineLimit(1)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - panels

struct Welcome: View {
    let processed: Int
    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "waveform").font(.system(size: 44)).foregroundStyle(.secondary)
            Text("Drop recordings here").font(.title3).bold()
            Text("Voice memos, .m4a, .mp3, .wav, or a video. Nothing starts until you "
                 + "press Transcribe. The audio never leaves this Mac.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)
            if processed > 0 {
                Text("\(processed) recordings already processed, listed in the sidebar.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

/// The pane for a recording that has not been processed yet.
///
/// The button sits in the same view as the sentence describing it. An earlier version
/// put the instruction here and the button in the sidebar, and a version after that
/// put both here inside a stack where the message claimed all the height and pushed
/// the button off the bottom of the window. Both read as a broken app.
struct PendingPanel: View {
    let item: QueueItem
    @Binding var language: String
    @Binding var expectedSpeakers: Int
    let languages: [(String, String)]
    let onStart: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.url.lastPathComponent).font(.title2).bold()
                    Text(item.url.deletingLastPathComponent().path)
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }

                GroupBox("Before starting") {
                    VStack(alignment: .leading, spacing: 14) {
                        Picker("Language", selection: $language) {
                            ForEach(languages, id: \.0) { Text($0.1).tag($0.0) }
                        }
                        Text("Pick the wrong language and Whisper does not fail. It guesses, "
                             + "and between two close languages the guess comes out as fluent, "
                             + "punctuated, invented text. When in doubt leave it automatic.")
                            .font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)

                        Divider()

                        Stepper(expectedSpeakers == 0
                                ? "People in the room: as many as it finds"
                                : "People in the room: \(expectedSpeakers)",
                                value: $expectedSpeakers, in: 0...12)
                        Text("The single setting that most affects whether the voices come "
                             + "out right. If you know the number, say it.")
                            .font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(6)
                }

                Button(action: onStart) {
                    Label("Transcribe", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                .disabled(item.state == .running)

                Text("Transcription runs on the CPU for about as long as the recording "
                     + "lasts, because the engine underneath has no Metal support. "
                     + "Separating the voices does run on Metal and takes a fraction of "
                     + "that. You can close the window; the work carries on.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(28)
            .frame(maxWidth: 620, alignment: .leading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }
}

/// The pane for a recording processed in an earlier session.
struct JobPanel: View {
    let job: JobSummary
    let loading: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(job.source).font(.title2).bold()
            HStack(spacing: 14) {
                if !job.recorded.isEmpty { Label(job.recorded, systemImage: "calendar") }
                if job.duration > 0 {
                    Label("\(Int(job.duration / 60)) min", systemImage: "clock")
                }
                Label(job.label, systemImage: "waveform")
            }
            .font(.callout).foregroundStyle(.secondary)

            if loading {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Reading this job…").foregroundStyle(.secondary)
                }
            } else if !job.isFinished {
                Text("This recording has been through part of the pipeline. Add it again "
                     + "to finish it: the stages already done are reused.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Button {
                NSWorkspace.shared.activateFileViewerSelecting(
                    [URL(fileURLWithPath: job.jobDir + "/output")])
            } label: {
                Label("Show the files", systemImage: "folder")
            }
            .disabled(!job.hasOutput)

            Spacer()
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

struct ProgressPanel: View {
    let phase: Phase
    let log: [String]
    let current: String
    let remaining: Int

    private let order: [Phase] = [.preparing, .detecting, .transcribing,
                                  .aligning, .diarizing, .identifying, .exporting]

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text(phase.rawValue).font(.title2).bold()
                if !current.isEmpty {
                    Text(current).font(.callout).foregroundStyle(.secondary)
                }
                if remaining > 0 {
                    Text("\(remaining) more after this one")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            ProgressView().progressViewStyle(.linear)

            VStack(alignment: .leading, spacing: 9) {
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

            // The last engine line, so a long stage still looks alive. Whisper prints a
            // percentage as it goes, and that is the only proof on screen that the
            // machine is busy rather than stuck.
            if let last = log.last {
                Text(last).font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(2).textSelection(.enabled)
            }
            Spacer()
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
