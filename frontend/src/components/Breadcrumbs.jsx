export default function Breadcrumbs({ parts }) {
  return (
    <div className="crumbs">
      {parts.map((p, i) => (
        <span key={i} style={{ display: "contents" }}>
          {i > 0 && <span className="sep">/</span>}
          {p.action ? <button onClick={p.action}>{p.label}</button> : <span>{p.label}</span>}
        </span>
      ))}
    </div>
  );
}
