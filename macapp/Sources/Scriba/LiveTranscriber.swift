import AVFoundation
import Foundation
import Speech

/// Live text while the microphone is running, from the speech model built into
/// macOS 26.
///
/// This is a preview and nothing more. It is shown while recording and thrown
/// away on stop; the document still comes from whisper large-v3, forced
/// alignment, pyannote and the voice registry, exactly as before. That is not
/// caution for its own sake, it is measured: on FLEURS Italian, 48 samples,
/// this model gets 3.87% of words wrong against a human reference and large-v3
/// gets 2.32%. On one real conversation the two disagree on 20% of words. Good
/// enough to watch, not good enough to keep.
///
/// What it does give, which whisper cannot: text about a second behind the
/// speaker, with punctuation and capitals already in it, for ten megabytes of
/// memory. Measured on this machine over a 405-second session: first hypothesis
/// 0.91 s after speech starts, committed text 1.03 s on average after an
/// utterance ends, memory flat throughout.
///
/// `.fastResults` is not optional. Without it the committed text arrives 3.3 s
/// late, which reads as a hang rather than as a transcript.
@available(macOS 26, *)
final class LiveTranscriber: LiveTranscribing, @unchecked Sendable {
    /// A finished utterance: the model will not revise this.
    var onFinal: ((String) -> Void)?
    /// The current hypothesis, which it will revise, usually within a second.
    var onGuess: ((String) -> Void)?

    private let transcriber: SpeechTranscriber
    private let analyzer: SpeechAnalyzer
    private let converter: AVAudioConverter
    private let analyzerFormat: AVAudioFormat
    private let continuation: AsyncStream<AnalyzerInput>.Continuation
    private var collector: Task<Void, Never>?
    private var finished = false

    /// Throws rather than degrading quietly: a recording that silently has no
    /// live text looks identical to a microphone that is not working.
    init(language: String, from tapFormat: AVAudioFormat) async throws {
        let locale = try await Self.resolve(language)

        transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [.volatileResults, .fastResults],
            attributeOptions: [.audioTimeRange])

        // The model is a system asset shared between applications. The first run
        // for a language downloads it; every run after finds it in place. Asking
        // even when it is already installed is deliberate: the inventory reported
        // an installed locale and still returned a request object.
        if let request = try? await AssetInventory.assetInstallationRequest(
            supporting: [transcriber]) {
            try await request.downloadAndInstall()
        }

        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(
            compatibleWith: [transcriber]) else {
            throw LiveError.noFormat
        }
        analyzerFormat = format

        // The microphone hands over 48 kHz float; the analyzer wants its own
        // format. Pushing the tap buffer in unconverted is not an error that can
        // be caught, it takes the process down inside the audio layer.
        guard let converter = AVAudioConverter(from: tapFormat, to: format) else {
            throw LiveError.noConverter
        }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        self.converter = converter

        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        self.continuation = continuation
        analyzer = SpeechAnalyzer(modules: [transcriber])
        try await analyzer.prepareToAnalyze(in: format)
        try await analyzer.start(inputSequence: stream)

        collector = Task { [weak self] in
            guard let self else { return }
            // Read as fast as results arrive. A slow reader here is not free: a
            // consumer that took 500 ms per result pushed the displayed text from
            // 3 s behind to 7 s behind, and the gap grew for the whole session.
            // Everything downstream of these callbacks is coalesced instead.
            do {
                for try await result in self.transcriber.results {
                    let text = String(result.text.characters)
                    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    else { continue }
                    if result.isFinal {
                        self.onFinal?(text)
                    } else {
                        self.onGuess?(text)
                    }
                }
            } catch {
                // The stream ends when the analyzer is finished, which is the
                // normal way out of this loop and not worth reporting.
            }
        }
    }

    /// Called from the audio thread. Converts and hands over, nothing else.
    func feed(_ buffer: AVAudioPCMBuffer) {
        guard !finished,
              let converted = Recorder.convert(buffer, with: converter, to: analyzerFormat)
        else { return }
        continuation.yield(AnalyzerInput(buffer: converted))
    }

    func finish() async {
        guard !finished else { return }
        finished = true
        continuation.finish()
        try? await analyzer.finalizeAndFinishThroughEndOfInput()
        collector?.cancel()
        collector = nil
    }

    /// The locale to ask for, from the language the window is set to.
    ///
    /// There is no language detection on this path. The pipeline decides the
    /// language of a recording by sampling five windows of the finished file with
    /// whisper, which cannot happen while the audio is still being spoken. So the
    /// preview uses what the picker says, and when the picker says "detect
    /// automatically" it uses the system language and the pane says it is a guess.
    private static func resolve(_ code: String) async throws -> Locale {
        let supported = await SpeechTranscriber.supportedLocales
        let wanted = (code == "auto" ? (Locale.current.language.languageCode?.identifier ?? "en") : code)
            .lowercased().replacingOccurrences(of: "_", with: "-")

        func norm(_ l: Locale) -> String {
            l.identifier.replacingOccurrences(of: "_", with: "-").lowercased()
        }
        if let exact = supported.first(where: { norm($0) == wanted }) { return exact }

        // A bare language code has to pick a region, and alphabetical order picks
        // badly: "en" lands on en-ZA. Prefer the region that shares the language's
        // own name, then the usual large ones.
        let sameLanguage = supported.filter {
            $0.language.languageCode?.identifier.lowercased() == wanted
        }
        if let home = sameLanguage.first(where: { norm($0) == wanted + "-" + wanted }) { return home }
        if let big = sameLanguage.first(where: {
            norm($0).hasSuffix("-us") || norm($0).hasSuffix("-gb")
        }) { return big }
        if let any = sameLanguage.sorted(by: { norm($0) < norm($1) }).first { return any }
        throw LiveError.noModel(code)
    }

    /// LocalizedError, not a bare Error: the pane shows `localizedDescription`,
    /// and a plain enum renders there as "The operation couldn't be completed."
    enum LiveError: LocalizedError {
        case noFormat
        case noConverter
        case noModel(String)

        var errorDescription: String? { message }

        var message: String {
            switch self {
            case .noFormat:
                return "macOS did not offer an audio format for live transcription."
            case .noConverter:
                return "The microphone format could not be converted for the speech model."
            case .noModel(let code):
                return "macOS has no on-device speech model for \(code). "
                     + "Recording still works; the document is unaffected."
            }
        }
    }
}
