import React, { useState } from 'react';
import './App.css';

function App() {
  const [pdf, setPdf] = useState(null);
  const [xray, setXray] = useState(null);
  const [eyeImage, setEyeImage] = useState(null);
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');

  const handleFileChange = (setter) => (e) => {
    setter(e.target.files[0]);
  };

  const handleQuerySubmit = () => {
    setResponse(`Analyzing: "${query}" with uploaded files...`);
  };

  return (
    <div className="container py-5 bg-light">
      {/* Header with Stethoscope SVG */}
      <div className="text-center mb-4">
        <img
          src="/images/statoscope.png"
          alt="Stethoscope"
          width="60"
          height="60"
          className="mb-3"
          style={{ objectFit: 'contain' }}
        />
        <h1 className="text-success mt-2">Automated Clinical Decision System</h1>
        <p className="text-muted">
          Upload clinical data and ask questions to receive AI-powered suggestions.
        </p>
      </div>


      {/* File Uploads */}
      <div className="card shadow-sm mb-4">
        <div className="card-body">
          <div className="mb-3">
            <label className="form-label text-success">Upload EHR Report (PDF)</label>
            <input type="file" accept="application/pdf" onChange={handleFileChange(setPdf)} className="form-control" />
          </div>

          <div className="mb-3">
            <label className="form-label text-success">Upload X-Ray Image</label>
            <input type="file" accept="image/*" onChange={handleFileChange(setXray)} className="form-control" />
          </div>

          <div className="mb-3">
            <label className="form-label text-success">Upload Eye Image</label>
            <input type="file" accept="image/*" onChange={handleFileChange(setEyeImage)} className="form-control" />
          </div>
        </div>
      </div>

      {/* Clinical Query Input */}
      <div className="card shadow-sm mb-4">
        <div className="card-body">
          <label className="form-label text-success">Enter Clinical Query</label>
          <textarea
            className="form-control mb-3"
            rows="4"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., What diagnosis fits this report?"
          ></textarea>
          <button onClick={handleQuerySubmit} className="btn btn-success w-100">
            🔍 Analyze & Respond
          </button>
        </div>
      </div>

      {/* Response Section */}
      <div className="card shadow-sm">
        <div className="card-body">
          <h5 className="text-success mb-3">Response</h5>
          <div className="border rounded p-3 bg-white" style={{ whiteSpace: 'pre-line' }}>
            {response || 'Your response will appear here after you submit a query.'}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
