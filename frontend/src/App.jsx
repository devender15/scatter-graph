import { Routes, Route } from 'react-router-dom';
import Hero from "./pages/Hero";
import Tool from "./pages/Tool";

function App() {

  return (
    <Routes>
      <Route path="/" element={<Hero />}/>
      <Route path="/tool" element={<Tool />}/>
    </Routes>
  );
}

export default App;
