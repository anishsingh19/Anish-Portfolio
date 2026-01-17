/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html", // Path to your main HTML file
    "./script.js",  // Include script.js if you add/remove Tailwind classes dynamically
    // You can add other paths here if you have more HTML files or template engines
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
