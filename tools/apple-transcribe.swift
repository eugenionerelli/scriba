// Transcribe an audio file with Apple's on-device SpeechTranscriber and print JSON.
//
// Why this exists: ctranslate2, the engine under faster-whisper, has no Metal
// backend, so on Apple Silicon whisper runs on the CPU and takes about as long as
// the recording lasts. Apple's own transcriber runs on the Neural Engine. Measured
// on an M4 against 6m45s of Spanish conversation: 443 seconds against 3.
//
// It is deliberately small and deliberately dumb. It reads a file, writes JSON on
// stdout, and knows nothing about scriba. The Python side decides what to do with
// the text, and the word timings it emits here are NOT used: they are contiguous,
// each word starting where the last one ended, so 124 of 182 seconds of silence in
// that same recording were swallowed into the words. scriba runs its own forced
// aligner over this text, which is what gives the real boundaries.
//
// Needs macOS 26. `swiftc -O apple-transcribe.swift -o apple-transcribe`.

import AVFoundation
import Foundation
import Speech

struct Line: Codable {
    let start: Double
    let end: Double
    let text: String
    let confidence: Double?
}

struct Output: Codable {
    let language: String
    let duration: Double
    let segments: [Line]
}

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

/// The locale to ask for, from a plain language code.
///
/// scriba speaks in ISO codes ("es"), the Speech framework wants a locale
/// ("es-ES"). Where several regions exist the choice does not matter much: the
/// model asset is per language, and asking for es-ES installs Chile, Mexico and
/// the United States with it.
func resolve(_ code: String) async -> Locale? {
    let supported = await SpeechTranscriber.supportedLocales
    let wanted = code.lowercased().replacingOccurrences(of: "_", with: "-")

    func norm(_ l: Locale) -> String {
        l.identifier.replacingOccurrences(of: "_", with: "-").lowercased()
    }
    if let exact = supported.first(where: { norm($0) == wanted }) { return exact }

    // A bare language code has to pick a region. Alphabetical order would give
    // es-CL for Spanish, which is a strange thing to label an Iberian recording
    // even though the model asset is the same for the whole language. Prefer the
    // region that shares the language's own name.
    let sameLanguage = supported.filter {
        $0.language.languageCode?.identifier.lowercased() == wanted
    }
    let home = wanted + "-" + wanted
    return sameLanguage.first(where: { norm($0) == home })
        ?? sameLanguage.first(where: { norm($0).hasSuffix("-us") || norm($0).hasSuffix("-gb") })
        ?? sameLanguage.sorted { norm($0) < norm($1) }.first
}

@main
struct Main {
    static func main() async {
        var args = Array(CommandLine.arguments.dropFirst())

        if args.first == "--locales" {
            let all = await SpeechTranscriber.supportedLocales
            print(all.map(\.identifier).sorted().joined(separator: "\n"))
            exit(0)
        }

        var language = "en"
        if let i = args.firstIndex(of: "--lang"), i + 1 < args.count {
            language = args[i + 1]
            args.removeSubrange(i...(i + 1))
        }
        guard let path = args.first else {
            fail("usage: apple-transcribe [--lang es] file.wav   |   --locales")
        }

        guard let locale = await resolve(language) else {
            let all = await SpeechTranscriber.supportedLocales.map(\.identifier).sorted()
            fail("no on-device model for language '\(language)'. Available: \(all.joined(separator: ", "))")
        }

        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence])

        // The model is a system asset, shared between applications and not part of
        // this program. The first run for a language downloads a few hundred
        // megabytes; every run after that finds it in place.
        if let request = try? await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            do { try await request.downloadAndInstall() }
            catch { fail("could not install the \(locale.identifier) speech model: \(error)") }
        }

        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else { fail("no such file: \(path)") }

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        var lines: [Line] = []

        let collector = Task {
            for try await result in transcriber.results where result.isFinal {
                let text = String(result.text.characters)
                guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
                // One confidence for the run, averaged over the words it contains.
                // A line the model was unsure of is worth marking, and whisper
                // gives nothing comparable.
                var scores: [Double] = []
                for run in result.text.runs {
                    if let c = run.transcriptionConfidence { scores.append(Double(c)) }
                }
                lines.append(Line(
                    start: result.range.start.seconds,
                    end: result.range.end.seconds,
                    text: text,
                    confidence: scores.isEmpty ? nil : scores.reduce(0, +) / Double(scores.count)))
            }
        }

        do {
            let file = try AVAudioFile(forReading: url)
            let duration = Double(file.length) / file.processingFormat.sampleRate
            if let last = try await analyzer.analyzeSequence(from: file) {
                try await analyzer.finalizeAndFinish(through: last)
            } else {
                try await analyzer.finalizeAndFinishThroughEndOfInput()
            }
            _ = try await collector.value

            let out = Output(language: locale.identifier, duration: duration,
                             segments: lines.sorted { $0.start < $1.start })
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.withoutEscapingSlashes]
            let data = try encoder.encode(out)
            FileHandle.standardOutput.write(data)
        } catch {
            fail("transcription failed: \(error)")
        }
    }
}
