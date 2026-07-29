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
        .onAppear {
            store.reload()
            AppDelegate.onQuit = { [weak engine] in engine?.terminateChild() }
        }
        // Coming back to the window is the moment to look again. The same engine
        // is usable from a terminal, on purpose, so a transcription can appear
        // while this window is open and it should not take a relaunch to see it.
        .onReceive(NotificationCenter.default.publisher(
            for: NSApplication.didBecomeActiveNotification)) { _ in
            if !engine.isRunning { store.reload() }
        }
        .onReceive(NotificationCenter.default.publisher(for: .scribaRefresh)) { _ in
            store.reload()
        }
        .onReceive(NotificationCenter.default.publisher(for: .scribaStart)) { _ in
            if !engine.isRunning { startAll() }
        }
        .onReceive(NotificationCenter.default.publisher(for: .scribaStop)) { _ in
            if engine.isRunning { stopEverything() }
        }
        .safeAreaInset(edge: .top) {
            // A wrong interpreter path used to look exactly like a clean install
            // with nothing in it: the state reader fails quietly and the sidebar
            // reads "Nothing yet". Say it once, at the top, before anyone presses
            // a button and waits for the failure.
            if !Engine.isConfigured {
                HStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("No Python interpreter at \(Engine.pythonPath)").bold()
                        Text("Open Settings and point it at the environment where "
                             + "scriba is installed. Nothing will run until then.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    SettingsLink { Text("Open Settings") }
                }
                .padding(12)
                .background(.regularMaterial)
            }
        }
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
                Button(action: openPanel) {
                    Label("Add recordings", systemImage: "plus")
                }
                .keyboardShortcut("o", modifiers: .command)
                .help("Add recordings to the queue")
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
                // Everything, including the jobs that produced nothing. Hiding
                // those meant the app could neither show the residue of a failed
                // run nor offer to clear it, while the terminal listed it plainly.
                ForEach(store.jobs) { job in
                    JobRowView(job: job).tag(Selection.job(job.jobDir))
                }
            }
        }
        .safeAreaInset(edge: .bottom) {
            if engine.isRunning {
                RunningStrip(phase: engine.phase, progress: engine.progress,
                             name: currentlyRunning,
                             remaining: queue.filter { $0.state == .waiting }.count,
                             onShow: { showRunning() },
                             onStop: stopEverything)
            } else if !queue.isEmpty {
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
        // The running job no longer takes the pane over. It used to, which meant
        // that for the length of a transcription, which is the length of the
        // recording, there was nothing else you could look at: not the transcript
        // you made this morning, not the one before it. The strip below the
        // sidebar says what is happening from wherever you are, and selecting the
        // running recording still gives you the full panel.
        if let running = queue.first(where: { $0.state == .running }),
           case .pending(let id) = selection, id == running.id {
            ProgressPanel(phase: engine.phase, line: engine.lastLine,
                          progress: engine.progress,
                          current: running.url.lastPathComponent,
                          remaining: queue.filter { $0.state == .waiting }.count,
                          onStop: stopEverything)
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

    /// Command O, because that is what it is on every other Mac application.
    private func openPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .movie]
        panel.allowsMultipleSelection = true
        panel.message = "Recordings to add to the queue. Nothing starts until you press Transcribe."
        if panel.runModal() == .OK { add(panel.urls) }
    }

    private func showRunning() {
        if let running = queue.first(where: { $0.state == .running }) {
            selection = .pending(running.id)
        }
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
                   maxSpeakers: expectedSpeakers > 0 ? expectedSpeakers : nil) { outcome in
            switch outcome {
            case .finished:
                mark(item.id, .finished)
            case .failed:
                // The reason, not the fact. "Did not finish" was true and useless,
                // and the alert carrying the real message is dismissed once.
                mark(item.id, .failed(engine.failureSummary))
            case .cancelled:
                // Asked for. Back in the queue rather than marked as broken.
                mark(item.id, .waiting)
            }
            store.reload()
            if runningAll, outcome != .cancelled { next() }
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

    /// Stop the running job and empty the rest of the queue.
    ///
    /// Stopping one and starting the next is not what anybody means by Stop, and
    /// the queue was started by one press of one button, so it ends the same way.
    /// Everything still waiting goes back to waiting rather than being thrown
    /// away: the files are still listed and one more press starts them again.
    private func stopEverything() {
        runningAll = false
        engine.cancel()
        for i in queue.indices where queue[i].state == .running {
            queue[i].state = .waiting
        }
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

/// What is happening, from wherever you are in the app.
///
/// Sits under the sidebar for the whole run, so the rest of the window is free
/// to be used. It says the stage in words, because a bar on its own tells you
/// something is moving and not what it is doing.
struct RunningStrip: View {
    let phase: Phase
    let progress: Double?
    let name: String
    let remaining: Int
    let onShow: () -> Void
    let onStop: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            Divider()
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(name).font(.callout).lineLimit(1)
                    Text(remaining > 0
                         ? "\(phase.rawValue.lowercased()), \(remaining) waiting"
                         : phase.rawValue)
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer(minLength: 6)
                Button(role: .destructive, action: onStop) {
                    Image(systemName: "stop.fill")
                }
                .help("Stop, and put the queue back to waiting")
            }
            if let progress {
                ProgressView(value: progress)
            } else {
                ProgressView().progressViewStyle(.linear)
            }
        }
        .padding(12)
        .background(.bar)
        .contentShape(Rectangle())
        .onTapGesture(perform: onShow)
        .help("Show the details of this run")
    }
}

struct ProgressPanel: View {
    let phase: Phase
    let line: String
    let progress: Double?
    let current: String
    let remaining: Int
    let onStop: () -> Void

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
            HStack(spacing: 14) {
                if let progress {
                    ProgressView(value: progress).progressViewStyle(.linear)
                    Text("\(Int(progress * 100))%")
                        .font(.system(.callout, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                } else {
                    ProgressView().progressViewStyle(.linear)
                }
                // The whole point of this app is that nothing starts without being
                // asked. Something that runs for an hour and cannot be called off
                // is the same problem wearing the opposite hat.
                Button(role: .destructive) { onStop() } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .controlSize(.regular)
            }

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
            if !line.isEmpty {
                Text(line).font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(2).textSelection(.enabled)
            }
            Spacer()
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
