import { useState } from "react";

import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import HomeScreen from "./screens/HomeScreen";
import Header, { HeaderPhone } from "./components/Header";

function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  return <BrowserRouter>
    <HeaderPhone menuOpen={menuOpen} setMenuOpen={setMenuOpen} />
    <Header menuOpen={menuOpen} setMenuOpen={setMenuOpen} />
    <Routes>
      <Route path="/" element={<HomeScreen />} />
    </Routes>
    <Toaster />
  </BrowserRouter>
}

export default App;

