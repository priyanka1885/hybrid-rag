/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef4ff",
          100: "#d9e6ff",
          500: "#2f6fed",
          600: "#1f57c9",
          700: "#1b45a0",
        },
      },
    },
  },
  plugins: [],
};
