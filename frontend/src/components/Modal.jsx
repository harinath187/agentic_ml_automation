import { useEffect, useRef } from "react";

export default function Modal({ title, description, onClose, children }) {
  const overlayRef = useRef(null);

  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div
      className="modal-overlay"
      ref={overlayRef}
      onClick={(e) => {
        if (e.target === overlayRef.current) onClose();
      }}
    >
      <div className="modal">
        <h3>{title}</h3>
        {description && <p className="desc">{description}</p>}
        {children}
      </div>
    </div>
  );
}
