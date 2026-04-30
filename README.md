# DRL for Combinatorial Optimization – IJCAI-ECAI 2026 Tutorial Website

This is the source code for the tutorial website accepted at the IJCAI-ECAI 2026 Tutorial Track.

## Files

```
├── index.html   ← Main page (all sections)
├── style.css    ← Styles and layout
├── main.js      ← Animated background + scroll effects
└── README.md    ← This file
```


## Adding Materials Later

When slides or notebooks are ready, update `index.html`:

- Find the `material-card--soon` cards
- Remove the `material-card--soon` class and the `material-badge` div
- Add an `href="YOUR-LINK"` attribute to the `<div>` → change it to `<a>`

## Customizing

- **Conference date/venue**: Edit the `.badge` text in `index.html`
- **Colors**: CSS variables are in `:root` in `style.css`
- **Fonts**: Google Fonts import in `<head>` of `index.html`
- **Paper link**: Search for `research.tue.nl` in `index.html` to update
