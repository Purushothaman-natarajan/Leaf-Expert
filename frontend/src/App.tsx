import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css';
import './components.css';
import { Navbar } from './components/Navbar';
import { HomePage } from './pages/HomePage';
import { AnalyzePage } from './pages/AnalyzePage';
import { TrainPage } from './pages/TrainPage';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/analyze" element={<AnalyzePage />} />
        <Route path="/train" element={<TrainPage />} />
      </Routes>
    </BrowserRouter>
  );
}
