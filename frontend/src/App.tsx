import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css';
import './components.css';
import { Navbar } from './components/Navbar';
import { HomePage } from './pages/HomePage';
import { AnalyzePage } from './pages/AnalyzePage';
import { TrainPage } from './pages/TrainPage';
import { QuickScanPage } from './pages/QuickScanPage';
import { DataBankPage } from './pages/DataBankPage';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/analyze" element={<AnalyzePage />} />
        <Route path="/train" element={<TrainPage />} />
        <Route path="/scan" element={<QuickScanPage />} />
        <Route path="/databank" element={<DataBankPage />} />
      </Routes>
    </BrowserRouter>
  );
}
