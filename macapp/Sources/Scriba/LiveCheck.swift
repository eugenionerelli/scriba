import AVFoundation
import Foundation

/// `Scriba --live-check recording.wav [--lang it]`
///
/// Puts an audio file through the same streaming path the microphone uses, at
/// wall-clock pace, and prints what came back and how late. It exists because
/// the live text is the one part of the application that cannot be tested
/// without sound: the unit tests cover the conversion, the file and the state
/// objects, and stop at the point where a speech model has to hear something.
///
/// Reading a file rather than opening the microphone is deliberate. It removes
/// the room, the permission and the person from the measurement, so the number
/// that comes out is the model and the plumbing rather than how loudly somebody
/// happened to speak. What it does not check is the microphone itself; for that
/// there is a person, a room, and the record button.
enum LiveCheck {

    static func run(path: String, language: String) -> Never {
        guard #available(macOS 26, *) else {
            print("live text needs macOS 26; this is \(ProcessInfo.processInfo.operatingSystemVersionString)")
            exit(1)
        }
        let url = URL(fileURLWithPath: path)
        guard let file = try? AVAudioFile(forReading: url) else {
            print("cannot read \(path)")
            exit(1)
        }
        let format = file.processingFormat
        let seconds = Double(file.length) / format.sampleRate
        print("file: \(Int(format.sampleRate)) Hz, \(format.channelCount) ch, "
              + String(format: "%.1f", seconds) + " s")
        print("language: \(language)")

        let done = DispatchSemaphore(value: 0)
        var failure: String?

        Task {
            let started = Date()
            var finals = 0, firstGuess: TimeInterval?, words = 0
            var lags: [TimeInterval] = []

            do {
                let live = try await LiveTranscriber(language: language, from: format)
                live.onGuess = { _ in
                    if firstGuess == nil { firstGuess = Date().timeIntervalSince(started) }
                }
                live.onFinal = { text in
                    finals += 1
                    words += text.split(separator: " ").count
                    // How far behind the audio the committed text arrived. The
                    // clock and the file are paced together, so wall time and
                    // audio position are the same quantity.
                    lags.append(Date().timeIntervalSince(started))
                }

                // 100 ms at a time, released on the clock, which is what the tap
                // does with a real microphone.
                let chunk = AVAudioFrameCount(format.sampleRate / 10)
                var index = 0
                while true {
                    guard let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                                        frameCapacity: chunk) else { break }
                    do { try file.read(into: buffer, frameCount: chunk) } catch { break }
                    if buffer.frameLength == 0 { break }
                    let due = started.addingTimeInterval(Double(index) * 0.1)
                    let wait = due.timeIntervalSinceNow
                    if wait > 0 { try? await Task.sleep(nanoseconds: UInt64(wait * 1e9)) }
                    live.feed(buffer)
                    index += 1
                }
                await live.finish()

                let elapsed = Date().timeIntervalSince(started)
                print(String(format: "first hypothesis: %.2f s after the audio started",
                             firstGuess ?? -1))
                print("committed utterances: \(finals), words: \(words)")
                if let last = lags.last {
                    print(String(format: "last committed text: %.2f s, audio ended at %.2f s, "
                                 + "so %.2f s behind", last, seconds, last - seconds))
                }
                print(String(format: "wall clock: %.1f s for %.1f s of audio", elapsed, seconds))
                if words == 0 {
                    failure = "no words came back: the model heard nothing in this file"
                }
            } catch {
                failure = "\(error.localizedDescription)"
            }
            done.signal()
        }

        done.wait()
        if let failure {
            print("FAILED: \(failure)")
            exit(2)
        }
        print("OK")
        exit(0)
    }
}
