import Foundation
import Security

/// The pyannote token, in the Keychain, under the same name the engine reads.
///
/// The Python side looks for a generic password with service `scriba-hf-token`
/// and takes the first thing it finds there (config.py, `keychain_get`). This
/// writes exactly that, so a token set here is the token the engine uses, and
/// there is one place it lives rather than two.
///
/// It goes through the Security framework rather than by running the `security`
/// command or `scriba token <value>`. Both of those put the secret in a process
/// argument list, where any other process on the machine can read it out of `ps`
/// for as long as the command runs. That is a strange way to handle a credential
/// whose whole point is not to be in a file.
enum Keychain {
    static let service = "scriba-hf-token"

    private static var account: String { NSUserName() }

    private static var query: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service]
    }

    /// Whether a token is there. Deliberately does not return it: nothing in this
    /// application needs to see the value, only whether one has been set.
    static func hasToken() -> Bool {
        var q = query
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        return SecItemCopyMatching(q as CFDictionary, nil) == errSecSuccess
    }

    /// Store, replacing whatever was there. Returns nil on success, a message otherwise.
    static func save(_ token: String) -> String? {
        let value = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return "The token is empty." }
        guard let data = value.data(using: .utf8) else { return "The token is not text." }

        // Delete first rather than update. An update has to match the existing
        // item's attributes exactly, and an item written by the `security` command
        // does not necessarily carry the same ones as an item written here; the
        // update then succeeds against nothing and the old token survives.
        SecItemDelete(query as CFDictionary)

        var item = query
        item[kSecAttrAccount as String] = account
        item[kSecValueData as String] = data
        // The engine runs as a separate process launched by this application and
        // has to read this while the machine is unlocked, which is the same
        // condition the command-line tool writes under.
        item[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlocked

        let status = SecItemAdd(item as CFDictionary, nil)
        guard status == errSecSuccess else {
            return SecCopyErrorMessageString(status, nil) as String?
                ?? "The Keychain refused it (error \(status))."
        }
        return nil
    }

    static func forget() {
        SecItemDelete(query as CFDictionary)
    }
}
