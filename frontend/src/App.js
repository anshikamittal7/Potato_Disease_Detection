import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import HomeScreen from "./screens/HomeScreen";


function App() {
  return <BrowserRouter>
    <Routes>
      <Route path="/" element={<HomeScreen />} />
    </Routes>
    <Toaster/>
  </BrowserRouter>
}

export default App;

