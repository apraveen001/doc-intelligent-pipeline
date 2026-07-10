Vite requires `index.html` to live at the project root, not inside `public/` —
the root `index.html` is the actual app entry point and imports `/src/main.jsx`.
This `public/` folder is for static assets served as-is (favicons, images, etc.)
and is otherwise empty for now.
