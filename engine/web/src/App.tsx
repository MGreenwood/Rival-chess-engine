import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import TrainModel from './pages/TrainModel';
import CommunityChallenge from './pages/CommunityChallenge';
import useStore from './store/store';

const App: React.FC = () => {
  const { theme } = useStore();

  return (
    <Router>
      <div className={theme.mode === 'dark' ? 'dark' : ''}>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/train" element={<TrainModel />} />
            <Route path="/community" element={<CommunityChallenge />} />
          </Routes>
        </Layout>
      </div>
    </Router>
  );
};

export default App;
