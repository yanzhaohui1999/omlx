// Regression coverage for #1929 — the blocking "Menubar Icon Hidden" alert
// that appeared on every launch for users who fold the oMLX icon away with
// Hidden Bar.
//
// `showHiddenAlert()` suppresses itself when a known menu bar manager is
// running, but the bundle-id list only covered Bartender / Ice / Thaw.
// Hidden Bar ships as `com.dwarvesv.minimalbar` (upstream `dwarvesf/hidden`),
// so its users hit the alert every single launch even though the icon was
// approved and merely hidden.
//
// `runningMenuBarManager()` reads `NSRunningApplication` and can't be driven
// from a unit test, so these pin the identifier list it iterates — which is
// the part that was wrong.

import XCTest
@testable import oMLX

final class MenubarVisibilityWatcherTests: XCTestCase {

    /// Hidden Bar — `dwarvesf/hidden`, `Hidden Bar.xcodeproj`, app target.
    @MainActor
    func testRecognisesHiddenBar() {
        XCTAssertTrue(
            MenubarVisibilityWatcher.menuBarManagerBundleIDs
                .contains("com.dwarvesv.minimalbar")
        )
    }

    /// Guards the managers that already worked; adding to the list must not
    /// quietly displace them.
    @MainActor
    func testStillRecognisesPreviouslySupportedManagers() {
        let ids = Set(MenubarVisibilityWatcher.menuBarManagerBundleIDs)
        for expected in [
            "com.surteesstudios.Bartender",
            "com.jordanbaird.Ice",
            "com.jordanbaird.ice",
            "com.stonerl.Thaw"
        ] {
            XCTAssertTrue(ids.contains(expected), "menu bar manager id dropped: \(expected)")
        }
    }

    /// `runningMenuBarManager()` returns the first match, so a repeated id
    /// costs an extra `NSRunningApplication` scan for no gain. Ice's two
    /// spellings differ only in case and are intentionally both present.
    @MainActor
    func testIdentifiersAreUnique() {
        let ids = MenubarVisibilityWatcher.menuBarManagerBundleIDs
        XCTAssertEqual(ids.count, Set(ids).count)
    }
}