// Ensure internal app links open in the same tab and use SPA navigation where possible
// This script intercepts clicks on anchors pointing to /player/ or /team/ and
// navigates using window.location (same tab) instead of letting the browser
// potentially open a new tab (some markdown renderers set target="_blank").

document.addEventListener('click', function (ev) {
    // Only handle left-clicks without modifier keys
    if (ev.defaultPrevented) return;
    if (ev.button !== 0) return; // left click
    if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;

    // Find the nearest anchor element
    let el = ev.target;
    while (el && el.tagName !== 'A') {
        el = el.parentElement;
    }
    if (!el || !el.href) return;

    try {
        // Only intercept same-origin internal links to player/team pages
        const url = new URL(el.href, window.location.origin);
        const path = url.pathname || '';
        if (path.startsWith('/player/') || path.startsWith('/team/')) {
            ev.preventDefault();
            // Use pushState to update the URL without a full reload, then dispatch an event
            // so Dash's router/clientside callbacks can react. We use location.assign to
            // force the same-tab navigation (assign vs replace keeps history).
            window.location.assign(url.href);
        }
    } catch (e) {
        // ignore malformed urls
    }
}, true);
