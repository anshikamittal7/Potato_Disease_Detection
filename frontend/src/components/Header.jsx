import React from "react";
import { AiOutlineMenu } from "react-icons/ai";

export const HeaderPhone = ({ menuOpen, setMenuOpen }) => {
  return (
    <div className={`navPhone ${menuOpen ? "navPhoneComes" : ""}`}>
      <NavContent setMenuOpen={setMenuOpen} />
    </div>
  );
};

const Header = ({ menuOpen, setMenuOpen }) => {
  return (
    <nav>
      <NavContent setMenuOpen={setMenuOpen} />
      <button className="navBtn" onClick={() => setMenuOpen(!menuOpen)}>
        <AiOutlineMenu />
      </button>
    </nav>
  );
};

const NavContent = ({ setMenuOpen }) => {
  return (
    <>
      <h2>PlantScan</h2>
      <div className="items">
        <a onClick={() => setMenuOpen(false)} href="/history">
          History
        </a>
        <a onClick={() => setMenuOpen(false)} href="/contact">
          Contact
        </a>
      </div>
    </>
  );
};

export default Header;
