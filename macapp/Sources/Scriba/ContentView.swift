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
    @StateObject private var recorder = Recorder()

    @State private var queue: [QueueItem] = []
    @State private var selection: Selection?
    @State private var language = "auto"
    @State private var expectedSpeakers = 0
    @State private var enrollOnSave = true
    @State private var isTargeted = false
    @State private var runningAll = false
    @State private var filter = ""

    enum Selection: Hashable {
        case recording
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
            AppDelegate.onQuit = { [weak engine, weak recorder] in
                engine?.terminateChild()
                // Quitting mid-recording has to close the file. The WAV header
                // carries the length, and it is written when the file is closed:
                // a recording left open is one every tool reads as zero seconds.
                if let recorder, recorder.isRecording {
                    let group = DispatchGroup()
                    group.enter()
                    Task { @MainActor in
                        await recorder.stop()
                        group.leave()
                    }
                    _ = group.wait(timeout: .now() + 3)
                }
            }
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
        .onReceive(NotificationCenter.default.publisher(for: .scribaRecord)) { _ in
            toggleRecording()
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(action: toggleRecording) {
                    Label(recorder.isRecording ? "Stop recording" : "Record",
                          systemImage: recorder.isRecording ? "stop.circle.fill" : "record.circle")
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .tint(recorder.isRecording ? .red : nil)
                .help(recorder.isRecording
                      ? "Stop and send it to be transcribed"
                      : "Record a conversation straight into scriba")
            }
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
            if recorder.isRecording {
                Section {
                    RecordingRow(meter: recorder.meter).tag(Selection.recording)
                }
            }
            if !queue.isEmpty {
                Section("Waiting to be transcribed") {
                    ForEach(queue) { item in
                        QueueRow(item: item).tag(Selection.pending(item.id))
                    }
                    .onDelete { offsets in
                        // Never silently drop the one that is running.
                        let doomed = offsets.map { queue[$0].id }
                        queue.removeAll { doomed.contains($0.id) && $0.state != .running }
                    }
                }
            }

            if let problem = store.problem {
                Section {
                    Label(problem, systemImage: "exclamationmark.triangle.fill")
                        .font(.callout)
                        .foregroundStyle(.orange)
                }
            }

            // Two groups, because they are two different things to a reader. One
            // has a document waiting; the other is a folder of intermediate files
            // and a reason it stopped. They used to sit in one list called
            // "Processed", distinguished by an icon, so the only way to find out
            // which was which was to click and see.
            group("Ready to read", jobs: ready, empty: emptyReadyText)
            group("Not finished", jobs: unfinished, empty: nil)
        }
        .searchable(text: $filter, placement: .sidebar,
                    prompt: "Recording or person")
    }

    @ViewBuilder
    private func group(_ title: String, jobs: [JobSummary], empty: String?) -> some View {
        if !jobs.isEmpty {
            Section("\(title) (\(jobs.count))") {
                ForEach(jobs) { job in
                    JobRowView(job: job).tag(Selection.job(job.jobDir))
                }
            }
        } else if let empty {
            Section(title) {
                Text(empty).font(.callout).foregroundStyle(.secondary)
            }
        }
    }

    private var emptyReadyText: String? {
        store.isLoading ? "Reading the list…"
            : filter.isEmpty ? "Nothing yet. Drop a recording anywhere in this window."
            : "Nothing matches \(filter)."
    }

    /// The recordings with a document, newest first.
    private var ready: [JobSummary] { matching.filter(\.isFinished) }

    /// The ones that stopped somewhere along the way. Kept visible rather than
    /// hidden: the residue of a run that failed is exactly what somebody comes
    /// looking for, and it is also what takes up the disk.
    private var unfinished: [JobSummary] { matching.filter { !$0.isFinished } }

    /// Filter on the name of the recording and on the people in it, because
    /// "the one with Ada in it" is how anybody actually looks for a conversation.
    private var matching: [JobSummary] {
        guard !filter.isEmpty else { return store.jobs }
        let needle = filter.lowercased()
        return store.jobs.filter { job in
            job.source.lowercased().contains(needle)
                || job.names.values.contains { $0.lowercased().contains(needle) }
        }
    }

    /// Say how long it will take before it starts, not after. Transcription runs for
    /// roughly the length of the recording, and somebody who does not know that reads
    /// a still progress bar as a hang.
    private var estimate: String {
        let waiting = queue.filter { $0.state == .waiting }
        guard !waiting.isEmpty else { return "" }
        let minutes = waiting.compactMap(\.minutes).reduce(0, +)
        guard minutes > 0 else { return "Runs for about as long as the recordings last." }
        return "About \(Int(minutes.rounded())) minutes of work, roughly the length of the audio."
    }

    /// Read the length once, off the main actor, and remember it on the item.
    private func measure(_ id: UUID, _ url: URL) {
        Task.detached(priority: .utility) {
            let asset = AVURLAsset(url: url)
            guard let duration = try? await asset.load(.duration) else { return }
            let seconds = CMTimeGetSeconds(duration)
            guard seconds.isFinite, seconds > 0 else { return }
            await MainActor.run {
                if let i = queueIndex(id) { queue[i].minutes = seconds / 60 }
            }
        }
    }

    private func queueIndex(_ id: UUID) -> Int? {
        queue.firstIndex(where: { $0.id == id })
    }

    // MARK: - detail

    @ViewBuilder
    private var detail: some View {
        if case .recording = selection, recorder.isRecording {
            RecordingPanel(recorder: recorder, language: $language,
                           languages: languages, onStop: toggleRecording)
        } else {
            transcriptionDetail
        }
    }

    @ViewBuilder
    private var transcriptionDetail: some View {
        // The running job no longer takes the pane over. It used to, which meant
        // that for the length of a transcription, which is the length of the
        // recording, there was nothing else you could look at: not the transcript
        // you made this morning, not the one before it. The strip below the
        // sidebar says what is happening from wherever you are, and selecting the
        // running recording still gives you the full panel.
        if let running = queue.first(where: { $0.state == .running }),
           case .pending(let id) = selection, id == running.id, !engine.isNaming {
            ProgressPanel(live: engine.live,
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
            Group {
                if !job.isFinished {
                    // A folder of intermediate files is not a transcript, and the
                    // speaker panel rendered for one is a heading with nothing
                    // under it and a button that cannot be pressed.
                    UnfinishedPanel(job: job, onFinish: { finish(job) })
                } else if let info = engine.info, info.jobDir == dir {
                    ResultPanel(engine: engine, info: info,
                                file: URL(fileURLWithPath: job.sourcePath),
                                enrollOnSave: $enrollOnSave)
                } else {
                    JobPanel(job: job, loading: engine.isLoadingState)
                }
            }
            // Keyed on the job, not on the view appearing. Clicking a second
            // recording keeps the same view in the same place, so onAppear does
            // not fire again: the title changed and the contents stayed on the
            // recording before it, which is worse than showing nothing.
            .task(id: job.jobDir) {
                engine.load(jobDir: job.jobDir, source: job.sourcePath)
            }
        } else {
            Welcome(processed: store.jobs.count, onRecord: toggleRecording)
        }
    }

    private var waitingCount: Int { queue.filter { $0.state == .waiting }.count }

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

    /// Move the selection onto the job the finished recording produced, and take
    /// the row out of the queue. Needs the list to have been re-read first.
    private func followFinished(_ item: QueueItem) {
        Task {
            await store.reloadAndWait()
            let path = item.url.path
            if let job = store.jobs.first(where: { $0.sourcePath == path }) {
                queue.removeAll { $0.id == item.id }
                selection = .job(job.jobDir)
            }
        }
    }

    /// Put an unfinished recording back in the queue, selected and ready.
    private func finish(_ job: JobSummary) {
        let url = URL(fileURLWithPath: job.sourcePath)
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        add([url])
        if let item = queue.first(where: { $0.url == url }) {
            selection = .pending(item.id)
        }
    }

    private func showRunning() {
        if let running = queue.first(where: { $0.state == .running }) {
            selection = .pending(running.id)
        }
    }

    // MARK: - actions

    /// Record, or stop and hand what was recorded to the queue.
    ///
    /// Stopping does not start the transcription. A recording that has just ended
    /// is the moment somebody is most likely to want to say how many people were
    /// in the room, or to record the next one straight away; a run that began on
    /// its own would hold the machine for as long as the conversation lasted.
    private func toggleRecording() {
        Task {
            if recorder.isRecording {
                if let url = await recorder.stop() {
                    add([url])
                    if let item = queue.first(where: { $0.url == url }) {
                        selection = .pending(item.id)
                    }
                } else {
                    selection = nil
                }
            } else {
                await recorder.start(language: language)
                if recorder.isRecording { selection = .recording }
            }
        }
    }

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
            let item = QueueItem(url: url)
            queue.append(item)
            measure(item.id, url)
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
                // Follow it into the list. The pane used to fall back to the
                // setup screen, offering to transcribe again a recording that
                // had just been transcribed, while the row sat under "Waiting".
                followFinished(item)
            case .failed:
                // The reason, not the fact. "Did not finish" was true and useless,
                // and the alert carrying the real message is dismissed once.
                mark(item.id, .failed(engine.failureSummary))
            case .cancelled:
                // Asked for. Back in the queue rather than marked as broken.
                mark(item.id, .waiting)
            }
            if case .finished = outcome {} else { store.reload() }
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

// MARK: - recording

/// The sidebar row while the microphone is open.
///
/// It observes `LiveMeter` and nothing else, on purpose. The level moves ten
/// times a second, and a view that observed the recorder as a whole would redraw
/// the sidebar, the queue and the job list at that rate.
struct RecordingRow: View {
    @ObservedObject var meter: LiveMeter

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "record.circle.fill")
                .foregroundStyle(.red)
                .symbolEffect(.pulse)
            VStack(alignment: .leading, spacing: 1) {
                Text("Recording").lineLimit(1)
                Text(elapsed).font(.caption).foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Spacer()
            LevelBar(peak: meter.peak)
        }
        .padding(.vertical, 2)
    }

    private var elapsed: String {
        let total = Int(meter.seconds)
        return String(format: "%02d:%02d", total / 60, total % 60)
    }
}

/// A level meter that exists to answer one question: is anything reaching the
/// microphone. A recording that comes out silent is otherwise invisible until
/// the transcript arrives empty half an hour later.
struct LevelBar: View {
    let peak: Float

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Capsule()
                    .fill(peak < 0.005 ? Color.orange : Color.green)
                    .frame(width: max(2, geo.size.width * CGFloat(scaled)))
            }
        }
        .frame(width: 44, height: 5)
        .help(peak < 0.005 ? "Nothing is reaching the microphone" : "Input level")
    }

    /// Ears are logarithmic and so are meters. On a linear scale ordinary speech
    /// sits in the leftmost tenth and the bar looks broken.
    private var scaled: Double {
        let db = 20 * log10(Double(max(peak, 1e-5)))
        return min(1, max(0, (db + 50) / 50))
    }
}

/// The pane while recording: how long, how loud, and the live text if it is on.
struct RecordingPanel: View {
    @ObservedObject var recorder: Recorder
    @Binding var language: String
    let languages: [(String, String)]
    let onStop: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            if recorder.liveTextEnabled {
                LiveTextView(live: recorder.live)
            } else {
                explanation
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                RecordingClock(meter: recorder.meter)
                Spacer()
                // Shown, not offered. The analyzer is built against the microphone
                // when the tap goes in, so this cannot change mid-session; it is
                // here to say which way it was set, and the places to set it are
                // the welcome screen and Settings.
                Label(recorder.liveTextEnabled ? "Live text on" : "Live text off",
                      systemImage: recorder.liveTextEnabled ? "text.bubble" : "text.bubble.badge.xmark")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .help("Set before recording starts, on the welcome screen or in Settings.")
                Button(role: .destructive, action: onStop) {
                    Label("Stop", systemImage: "stop.fill")
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
            }
            if let warning = recorder.warning {
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.orange)
            }
        }
        .padding(20)
    }

    private var explanation: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Recording to ~/.scriba/recordings")
                .font(.callout)
            Text("When you stop, the recording is added to the queue. Nothing is "
                 + "transcribed until you press Transcribe, because that runs for "
                 + "about as long as the conversation did.")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text("Turn on Live text before you start recording to see the words as "
                 + "they are spoken. That preview comes from the speech model built "
                 + "into macOS and is thrown away when you stop; the document is "
                 + "always produced afterwards by whisper large-v3, which makes "
                 + "fewer mistakes.")
                .font(.callout)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(20)
        .frame(maxWidth: 620, alignment: .leading)
    }
}

/// Split out so the ticking clock redraws by itself and not the whole panel.
struct RecordingClock: View {
    @ObservedObject var meter: LiveMeter

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "record.circle.fill")
                .font(.title2)
                .foregroundStyle(.red)
                .symbolEffect(.pulse)
            Text(elapsed).font(.system(.title, design: .monospaced))
            LevelBar(peak: meter.peak)
        }
    }

    private var elapsed: String {
        let total = Int(meter.seconds)
        return String(format: "%02d:%02d", total / 60, total % 60)
    }
}

/// The live preview. Committed text in the normal colour, the current hypothesis
/// greyed: the second half is still moving and the reader should be able to see
/// which half that is.
struct LiveTextView: View {
    @ObservedObject var live: LiveText

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    if let unavailable = live.unavailable {
                        Label(unavailable, systemImage: "info.circle")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                    // One Text built by concatenation rather than two views, so
                    // the hypothesis flows on from the settled words instead of
                    // starting its own paragraph. The modifier goes on the whole
                    // thing: applied to the first half it stops being a Text.
                    (Text(live.settled)
                     + Text(live.settled.isEmpty ? "" : " ")
                     + Text(live.guess).foregroundColor(.secondary))
                        .textSelection(.enabled)

                    if live.settled.isEmpty && live.guess.isEmpty && live.unavailable == nil {
                        Text("Listening…").foregroundStyle(.secondary)
                    }
                    Color.clear.frame(height: 1).id(bottom)
                }
                .font(.system(size: 15))
                .lineSpacing(4)
                .frame(maxWidth: 720, alignment: .leading)
                .padding(20)
            }
            .onChange(of: live.settled) { _, _ in
                withAnimation { proxy.scrollTo(bottom, anchor: .bottom) }
            }
        }
    }

    private var bottom: String { "live-bottom" }
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
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 1) {
                Text(job.source)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            // Without this the longest name in the list decides how wide the
            // column wants to be, and the sidebar grew until it ran off the
            // side of the screen.
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .help(job.names.isEmpty ? job.source
              : job.source + " · " + job.names.values.sorted().joined(separator: ", "))
    }

    /// One line, in the order somebody scans it: when, how long, who.
    private var subtitle: String {
        var parts: [String] = []
        if !job.recorded.isEmpty { parts.append(job.recorded) }
        if job.duration > 0 { parts.append("\(Int(job.duration / 60)) min") }
        if job.isFinished {
            if !job.names.isEmpty {
                parts.append(job.names.values.sorted().joined(separator: ", "))
            } else if job.speakers > 0 {
                parts.append(job.speakers == 1 ? "1 voice, unnamed"
                                               : "\(job.speakers) voices, unnamed")
            }
        } else {
            parts.append(job.label.lowercased())
        }
        return parts.joined(separator: "  ")
    }
}

// MARK: - panels

/// A recording that started and stopped somewhere in the middle.
///
/// Says which part exists and which does not, in the order the work happens, and
/// offers the one thing worth doing about it. The alternative, and what this
/// replaced, was the speaker panel with nothing in it: a heading called "Who is
/// speaking" over an empty space, and a save button that could not be pressed.
struct UnfinishedPanel: View {
    let job: JobSummary
    let onFinish: () -> Void

    private var sourceIsThere: Bool {
        FileManager.default.fileExists(atPath: job.sourcePath)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 6) {
                Text(job.source).font(.title2).bold().textSelection(.enabled)
                HStack(spacing: 14) {
                    if !job.recorded.isEmpty {
                        Label(job.recorded, systemImage: "calendar")
                    }
                    if job.duration > 0 {
                        Label(humanDuration(job.duration), systemImage: "clock")
                    }
                }
                .font(.callout).foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 10) {
                Text(job.label).font(.headline)
                Text(explanation).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }

            if sourceIsThere {
                Button(action: onFinish) {
                    Label("Put it back in the queue", systemImage: "play.fill")
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                Text("It picks up from what is already there. The stages it finished "
                     + "are not done again.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                Label("The recording is no longer at \(job.sourcePath)",
                      systemImage: "questionmark.folder")
                    .foregroundStyle(.orange)
                Text("Put the file back where it was, or drop it in again as a new "
                     + "recording. What was computed for it is still here.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Button {
                NSWorkspace.shared.activateFileViewerSelecting(
                    [URL(fileURLWithPath: job.jobDir)])
            } label: {
                Label("Show what it produced", systemImage: "folder")
            }
            .buttonStyle(.link)

            }
            .padding(28)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var explanation: String {
        switch job.state {
        case "transcribed":
            return "The words and the voices are both here. The documents were never "
                 + "written, which is the last and quickest step."
        case "voices only":
            return "The voices were separated and the speech was never transcribed. "
                 + "Transcription is the long part: it takes about as long as the "
                 + "recording lasts."
        case "text only":
            return "The words are here and nobody was separated, so there is a "
                 + "transcript with no idea of who said what."
        default:
            return "This run stopped before it produced anything. Nothing was kept "
                 + "except the folder."
        }
    }
}

struct Welcome: View {
    let processed: Int
    let onRecord: () -> Void
    /// The live preview has to be decided here, because once recording has started
    /// it is too late: the analyzer is built against the microphone at the moment
    /// the tap is installed. It used to live only in the recording panel, where
    /// the switch was visible and disabled and there was no way to reach it in
    /// time, which is a switch that does nothing.
    @AppStorage("liveTextEnabled") private var liveText = false

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

            Divider().frame(width: 260).padding(.vertical, 4)

            Button(action: onRecord) {
                Label("Record a conversation", systemImage: "record.circle")
            }
            .controlSize(.large)
            Toggle("Show the words while recording", isOn: $liveText)
                .toggleStyle(.checkbox)
            Text(liveText
                 ? "A preview from the model built into macOS, about a second behind. "
                   + "It is not kept: the document still comes from whisper afterwards."
                 : "The recording goes into the queue when you stop it.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)

            if processed > 0 {
                Text("\(processed) recordings already processed, listed in the sidebar.")
                    .font(.caption).foregroundStyle(.secondary)
                    .padding(.top, 6)
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
    @ObservedObject var live: RunState
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
                         ? "\(live.phase.rawValue.lowercased()), \(remaining) waiting"
                         : live.phase.rawValue)
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer(minLength: 6)
                Button(role: .destructive, action: onStop) {
                    Image(systemName: "stop.fill")
                }
                .help("Stop, and put the queue back to waiting")
            }
            if let progress = live.progress {
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
    @ObservedObject var live: RunState
    let current: String
    let remaining: Int
    let onStop: () -> Void

    private let order: [Phase] = [.preparing, .detecting, .transcribing,
                                  .aligning, .diarizing, .identifying, .exporting]

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text(live.phase.rawValue).font(.title2).bold()
                if !current.isEmpty {
                    Text(current).font(.callout).foregroundStyle(.secondary)
                }
                if remaining > 0 {
                    Text("\(remaining) more after this one")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 14) {
                if let progress = live.progress {
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
                    let cur = order.firstIndex(of: live.phase) ?? 0
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
            if !live.lastLine.isEmpty {
                Text(live.lastLine).font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(2).textSelection(.enabled)
            }
            Spacer()
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
