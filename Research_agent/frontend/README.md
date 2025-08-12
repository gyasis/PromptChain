# Research Agent Frontend

A modern, responsive web interface for the Research Agent system built with Svelte, TypeScript, and Tailwind CSS.

## Design System

The frontend follows a clean, modern design language inspired by contemporary academic and research platforms:

- **Primary Colors**: Orange/coral accent colors (#ff7733) for CTAs and highlighting
- **Neutral Palette**: Clean greys from #fafafa (backgrounds) to #0a0a0a (text)
- **Typography**: Inter font family for optimal readability
- **Rounded Corners**: Consistent use of rounded-xl (1rem) and rounded-2xl (1.5rem)
- **Soft Shadows**: Layered shadow system for depth without heaviness
- **Card-based Layout**: Content organized in clean, bordered cards

## Features

### Current Implementation
- **Dashboard View**: Overview of research metrics and quick start
- **Sessions Management**: Browse and manage research sessions
- **System Health**: Real-time backend connection status
- **Responsive Design**: Mobile-first approach with desktop optimization
- **Mock Data**: Demonstration data for development and testing

### Planned Features
- **Real-time Progress**: WebSocket integration for live research updates
- **Interactive Chat**: Q&A interface with research findings
- **Data Visualization**: Charts and graphs for research insights
- **Export Functions**: PDF, CSV, and JSON export capabilities
- **Session Details**: Drill-down views for individual research sessions

## Project Structure

```
frontend/
├── src/
│   ├── App.svelte          # Main application component
│   ├── app.css            # Tailwind + custom styles
│   ├── main.ts            # Application entry point
│   └── lib/               # Reusable components (future)
├── public/                # Static assets
├── package.json           # Dependencies and scripts
├── tailwind.config.js     # Tailwind configuration
├── postcss.config.js      # PostCSS configuration
├── vite.config.js         # Vite build configuration
└── tsconfig.*.json        # TypeScript configuration
```

## Setup Instructions

### Prerequisites
- Node.js 18+ 
- npm or pnpm
- Research Agent backend running on port 8078

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```
   The app will be available at `http://localhost:5173`

3. **Build for production:**
   ```bash
   npm run build
   ```

4. **Preview production build:**
   ```bash
   npm run preview
   ```

## Development

### CSS Architecture

The project uses a layered CSS approach:

1. **Base Layer**: Tailwind base styles + CSS custom properties
2. **Components Layer**: Reusable UI components (.card, .btn, .input, etc.)
3. **Utilities Layer**: Custom utility classes and animations

### Custom Component Classes

Key component classes available:

- **Cards**: `.card`, `.card-hover`, `.card-active`
- **Buttons**: `.btn-primary`, `.btn-secondary`, `.btn-ghost`, `.btn-outline`
- **Form Elements**: `.input`, `.textarea`
- **Status**: `.badge-*`, `.status-dot-*`, `.phase-*`
- **Layout**: `.section-header`, `.metric-card`, `.tier-card`

### State Management

Currently using Svelte's built-in reactivity with planned integration:

- **API Integration**: Axios for HTTP requests
- **WebSocket**: Native WebSocket API for real-time updates
- **Local Storage**: Session persistence (planned)

### Backend Integration

The frontend connects to the FastAPI backend at `http://localhost:8078/api` with endpoints:

- `GET /health` - System health check
- `GET /sessions` - List research sessions
- `POST /research/start` - Start new research
- `WebSocket /ws/progress/{session_id}` - Real-time progress
- `WebSocket /ws/chat/{session_id}` - Interactive chat

## Customization

### Theme Configuration

Update `tailwind.config.js` to modify:
- Colors (primary/neutral palettes)
- Font families
- Border radius values
- Shadow definitions
- Animation timings

### Component Styling

Modify `src/app.css` component layer for:
- Button variants
- Card styles
- Form elements
- Status indicators
- Layout components

## Performance

### Build Optimization
- Vite for fast builds and hot reload
- TypeScript for type safety
- Tree-shaking for minimal bundle size
- CSS purging for unused styles

### Runtime Performance
- Svelte's compile-time optimizations
- Minimal JavaScript bundle
- Efficient reactivity system
- Lazy loading for route splits (planned)

## Browser Support

- **Modern Browsers**: Chrome 88+, Firefox 78+, Safari 14+, Edge 88+
- **CSS Features**: CSS Grid, Flexbox, CSS Custom Properties
- **JavaScript**: ES2020 features, async/await, modules

## Contributing

1. Follow the existing component structure
2. Use Tailwind classes instead of custom CSS where possible
3. Maintain the design system consistency
4. Add TypeScript types for new components
5. Test responsiveness across breakpoints

## Deployment

### Static Hosting
The built application is a static SPA suitable for:
- Netlify
- Vercel
- GitHub Pages
- Any static web server

### Configuration
Update the `API_BASE` constant in `App.svelte` for production backend URL.

### Build Process
```bash
npm run build
# Output will be in dist/ directory
```
