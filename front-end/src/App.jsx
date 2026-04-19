import { useState, Component } from 'react';
import { RotateCcw } from 'lucide-react';
import { startFullPipeline } from './services/api';
import UploadZone from './components/UploadZone.jsx';
import ExecutionTerminal from './components/ExecutionTerminal.jsx';
import ResultsDashboard from './components/ResultsDashboard.jsx';
import './App.css';

const PAGES = ['Upload', 'Pipeline', 'Results'];

/* Error Boundary to catch crashes and show a message instead of a blank screen */
class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { hasError: false, error: null }; }
  static getDerivedStateFromError(error) { return { hasError: true, error }; }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '3rem', color: '#f87171', fontFamily: 'monospace' }}>
          <h2>Something crashed</h2>
          <pre style={{ marginTop: '1rem', color: '#ececef', whiteSpace: 'pre-wrap' }}>{this.state.error?.message}</pre>
          <button style={{ marginTop: '1rem', padding: '0.5rem 1rem', background: '#34d399', color: '#000', border: 'none', borderRadius: '6px', cursor: 'pointer' }}
            onClick={() => { this.setState({ hasError: false, error: null }); window.location.reload(); }}>
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [pipelineState, setPipelineState] = useState('idle');
  const [pipelinePayload, setPipelinePayload] = useState(null);
  const [finalResults, setFinalResults] = useState(null);
  const [runId, setRunId] = useState(null);

  const handleFileSelect = async (file, targetColumn) => {
    try {
      const payload = await startFullPipeline(file, targetColumn);
      setRunId(payload.run_id);
      setPipelinePayload(payload);
      setPipelineState('running');
    } catch (err) {
      alert("Failed to start pipeline: " + err.message);
    }
  };

  const handlePipelineComplete = (results) => {
    setFinalResults(results);
    setPipelineState('complete');
  };

  const handleReset = () => {
    setPipelineState('idle');
    setPipelinePayload(null);
    setFinalResults(null);
    setRunId(null);
  };

  const activeIdx = pipelineState === 'idle' ? 0 : pipelineState === 'running' ? 1 : 2;

  return (
    <ErrorBoundary>
      <div className="app-shell">
        {/* ── Header ─────────────────────────── */}
        <header className="app-header">
          <div className="app-logo">
            <span className="dot" />
            AutoEDA
          </div>

          <div className="header-nav">
            {PAGES.map((p, i) => (
              <span key={p} className={`nav-pill${i === activeIdx ? ' active' : ''}`}>{p}</span>
            ))}
          </div>

          <div className="header-actions">
            {pipelineState !== 'idle' && (
              <button className="btn btn-ghost btn-sm" onClick={handleReset}>
                <RotateCcw size={13} /> Reset
              </button>
            )}
          </div>
        </header>

        {/* ── Main ───────────────────────────── */}
        <main className="app-main">
          {pipelineState === 'idle' && <UploadZone onFileSelect={handleFileSelect} />}
          {pipelineState === 'running' && pipelinePayload && (
            <ExecutionTerminal initialPayload={pipelinePayload} onPipelineComplete={handlePipelineComplete} />
          )}
          {pipelineState === 'complete' && finalResults && (
            <ResultsDashboard results={finalResults} runId={runId} />
          )}
        </main>
      </div>
    </ErrorBoundary>
  );
}

export default App;
