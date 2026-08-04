import AVFoundation
import Foundation
import SwiftUI

/// Recording a conversation straight into scriba.
///
/// Until now the path from a conversation to a document went through the phone,
/// Voice Memos, an export to the Desktop and a drag into this window. The audio
/// this produces is an ordinary file in an ordinary folder, and once it is
/// stopped it goes through the same pipeline as anything dropped in: whisper,
/// alignment, pyannote, the voice registry. Nothing here shortcuts that.
///
/// Three objects rather than one, and the split is not cosmetic. `Recorder`
/// changes state a handful of times per session. `LiveMeter` changes about ten
/// times a second, and `LiveText` changes whenever the speech model has another
/// guess. SwiftUI redraws every view observing an object that changed, so
/// putting the level meter on the same object as everything else would redraw
/// the whole window ten times a second. Engine.swift carries a comment about
/// this exact mistake having been made once already.

// MARK: - level and elapsed time

/// The two numbers that move constantly while recording.
@MainActor final class LiveMeter: ObservableObject {
    @Published private(set) var peak: Float = 0
    @Published private(set) var seconds: Double = 0

    /// Written from the audio thread's callback, so it is coalesced: the tap
    /// fires about every 100 ms and the eye cannot read faster than that anyway.
    private var lastPublish = Date.distantPast

    func update(peak: Float, seconds: Double) {
        let now = Date()
        guard now.timeIntervalSince(lastPublish) >= 0.1 else { return }
        lastPublish = now
        self.peak = peak
        self.seconds = seconds
    }

    func reset() {
        peak = 0
        seconds = 0
        lastPublish = .distantPast
    }
}

/// What the live speech model thinks it has heard so far.
///
/// `settled` is what it has committed to; `guess` is the hypothesis for the
/// words being spoken right now, which it will revise. They are kept apart
/// because the interface shows them differently, and because a reader deserves
/// to know which half is still moving.
@MainActor final class LiveText: ObservableObject {
    @Published private(set) var settled: String = ""
    @Published private(set) var guess: String = ""
    @Published private(set) var unavailable: String?

    private var lastGuessPublish = Date.distantPast

    func commit(_ text: String) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }
        settled = settled.isEmpty ? clean : settled + " " + clean
        guess = ""
        lastGuessPublish = .distantPast
    }

    func hypothesise(_ text: String) {
        let now = Date()
        guard now.timeIntervalSince(lastGuessPublish) >= 0.1 else { return }
        lastGuessPublish = now
        guess = text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func note(unavailable reason: String?) { unavailable = reason }

    func reset() {
        settled = ""
        guess = ""
        unavailable = nil
        lastGuessPublish = .distantPast
    }
}

// MARK: - the recorder

@MainActor final class Recorder: ObservableObject {
    enum State: Equatable { case idle, recording, failed(String) }

    @Published private(set) var state: State = .idle
    /// Where the finished recording ended up, for the caller to hand to the engine.
    @Published private(set) var finished: URL?
    /// Set when a session produced silence: a common and otherwise invisible failure.
    @Published var warning: String?

    let meter = LiveMeter()
    let live = LiveText()

    /// Whether to show live text while recording. Off by default: it is a preview,
    /// it costs a language decision up front, and the document does not come from it.
    @AppStorage("liveTextEnabled") var liveTextEnabled = false

    static let folder = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".scriba/recordings")

    private var engine: AVAudioEngine?
    private var file: AVAudioFile?
    private var fileConverter: AVAudioConverter?
    private var fileFormat: AVAudioFormat?
    private var url: URL?
    private var started: Date?
    private var framesWritten: AVAudioFramePosition = 0
    private var loudestSeen: Float = 0
    private var transcriber: LiveTranscribing?

    var isRecording: Bool { state == .recording }

    // MARK: starting

    /// Ask for the microphone, then start. `language` is only used by the live
    /// preview: the document's language is still decided by the pipeline, which
    /// samples the whole recording rather than guessing from the first seconds.
    func start(language: String) async {
        guard state != .recording else { return }
        finished = nil
        warning = nil
        meter.reset()
        live.reset()
        loudestSeen = 0
        framesWritten = 0

        // Ask before arming anything. Starting the engine without the permission
        // does not fail: it records frames of digital silence, and the first sign
        // of trouble is a transcript of nothing half an hour later.
        guard await Self.microphoneGranted() else {
            state = .failed("Scriba cannot use the microphone. Give it permission in "
                            + "System Settings > Privacy & Security > Microphone, then try again.")
            return
        }

        do {
            try await beginSession(language: language)
            state = .recording
            started = Date()
        } catch {
            state = .failed(Self.readable(error))
        }
    }

    private static func microphoneGranted() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }

    private func beginSession(language: String) async throws {
        try FileManager.default.createDirectory(at: Self.folder, withIntermediateDirectories: true)

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let tapFormat = input.outputFormat(forBus: 0)
        guard tapFormat.sampleRate > 0 else {
            throw RecorderError.noInput
        }

        // 48 kHz mono 16-bit on disk. The pipeline resamples to 16 kHz itself and
        // applies loudnorm and a high-pass on the way, so writing anything fancier
        // here would only be thrown away; writing anything smaller would throw away
        // information before that filter got to see it.
        guard let fileFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                             sampleRate: 48_000, channels: 1,
                                             interleaved: true),
              let fileConverter = AVAudioConverter(from: tapFormat, to: fileFormat) else {
            throw RecorderError.noConverter
        }
        fileConverter.sampleRateConverterQuality = AVAudioQuality.max.rawValue

        let stamp = Self.stampFormatter.string(from: Date())
        let url = Self.folder.appendingPathComponent("Recording \(stamp).wav")
        let file = try AVAudioFile(forWriting: url, settings: fileFormat.settings,
                                   commonFormat: .pcmFormatInt16, interleaved: true)

        // The live preview is a second, independent consumer of the same tap. It
        // gets its own converter: the analyzer wants its own format, and feeding
        // one converter's output into a file expecting another format is a trap
        // door, not an error (it traps in the audio layer and takes the app down).
        var live: LiveTranscribing?
        if liveTextEnabled {
            if #available(macOS 26, *) {
                do {
                    let t = try await LiveTranscriber(language: language, from: tapFormat)
                    t.onFinal = { [weak self] text in
                        Task { @MainActor in self?.live.commit(text) }
                    }
                    t.onGuess = { [weak self] text in
                        Task { @MainActor in self?.live.hypothesise(text) }
                    }
                    live = t
                } catch {
                    // A missing speech model must not cost the recording. Say what
                    // happened in the pane and keep the microphone running.
                    self.live.note(unavailable: Self.readable(error))
                }
            } else {
                self.live.note(unavailable: "Live text needs macOS 26. Recording still works.")
            }
        }

        let meter = self.meter
        input.installTap(onBus: 0, bufferSize: 4800, format: tapFormat) { [weak self] buffer, _ in
            // This closure runs on the audio thread. Nothing here may block, take
            // a lock held by the main actor, or allocate unpredictably.
            let loudest = Self.peak(buffer)
            if let converted = Self.convert(buffer, with: fileConverter, to: fileFormat) {
                try? file.write(from: converted)
            }
            live?.feed(buffer)
            let seconds = Double(file.length) / fileFormat.sampleRate
            Task { @MainActor [weak self] in
                meter.update(peak: loudest, seconds: seconds)
                self?.loudestSeen = max(self?.loudestSeen ?? 0, loudest)
            }
        }

        try engine.start()
        self.engine = engine
        self.file = file
        self.fileConverter = fileConverter
        self.fileFormat = fileFormat
        self.url = url
        self.transcriber = live
    }

    // MARK: stopping

    /// Stop, close the file, and hand back where it is. Safe to call twice.
    @discardableResult
    func stop() async -> URL? {
        guard state == .recording || engine != nil else { return nil }

        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil

        await transcriber?.finish()
        transcriber = nil

        let length = file?.length ?? 0
        let rate = fileFormat?.sampleRate ?? 48_000
        // Closing the file is what writes the WAV header's real length. Dropping
        // the reference is the documented way to do it, and quitting without
        // doing it leaves a file whose duration every tool reads as unknown.
        file = nil
        fileConverter = nil

        state = .idle
        let seconds = Double(length) / rate
        guard let url, seconds >= 1.0 else {
            if let url { try? FileManager.default.removeItem(at: url) }
            self.url = nil
            warning = "That recording was under a second, so it was not kept."
            return nil
        }

        if loudestSeen < 0.001 {
            // Permission can be granted and the input still be the wrong device,
            // muted, or unplugged. The file exists and contains nothing.
            warning = "The recording came out silent: nothing reached the microphone. "
                    + "Check the input device in System Settings > Sound."
        }

        self.url = nil
        finished = url
        return url
    }

    // MARK: helpers

    private static let stampFormatter: DateFormatter = {
        let f = DateFormatter()
        // Sortable, and legible in a Finder window a month later. Colons are not
        // allowed in a file name on this filesystem, hence the dashes.
        f.dateFormat = "yyyy-MM-dd HH-mm"
        return f
    }()

    private static func peak(_ buffer: AVAudioPCMBuffer) -> Float {
        guard let channel = buffer.floatChannelData else { return 0 }
        var loudest: Float = 0
        for i in 0..<Int(buffer.frameLength) {
            loudest = max(loudest, abs(channel[0][i]))
        }
        return loudest
    }

    /// One converter for the whole session, never one per buffer: a fresh
    /// converter resets the resampler's filter state and the seam is audible.
    ///
    /// nonisolated because it is called from the audio thread. Hopping to the main
    /// actor to resample audio would be a glitch you could hear.
    nonisolated static func convert(_ input: AVAudioPCMBuffer, with converter: AVAudioConverter,
                        to format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let ratio = format.sampleRate / input.format.sampleRate
        let capacity = AVAudioFrameCount(Double(input.frameLength) * ratio) + 1024
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
            return nil
        }
        var handedOver = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, status in
            if handedOver { status.pointee = .noDataNow; return nil }
            handedOver = true
            status.pointee = .haveData
            return input
        }
        guard status != .error, output.frameLength > 0 else { return nil }
        return output
    }

    private static func readable(_ error: Error) -> String {
        if let e = error as? RecorderError { return e.message }
        return (error as NSError).localizedDescription
    }
}

enum RecorderError: Error {
    case noInput
    case noConverter

    var message: String {
        switch self {
        case .noInput:
            return "No audio input. Check the microphone in System Settings > Sound."
        case .noConverter:
            return "This machine's microphone format could not be converted to a "
                 + "recordable one. Please report the input device."
        }
    }
}

/// What the recorder needs from a live transcriber, so the recorder does not have
/// to be annotated for macOS 26 from top to bottom.
protocol LiveTranscribing: AnyObject {
    /// Called from the audio thread with the tap's own buffer.
    func feed(_ buffer: AVAudioPCMBuffer)
    func finish() async
}
