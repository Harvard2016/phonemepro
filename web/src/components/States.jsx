export function LoadingState({ label }) {
  return (
    <div className="state" role="status">
      <div className="state__pulse" aria-hidden="true" />
      <p className="state__text">{label}</p>
    </div>
  )
}

export function MessageState({ title, children, action }) {
  return (
    <div className="state">
      <p className="state__title">{title}</p>
      {children && <p className="state__text">{children}</p>}
      {action}
    </div>
  )
}
