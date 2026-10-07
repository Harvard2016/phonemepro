import { Component } from 'react'
import { MessageState } from './components/States'

class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  render() {
    if (this.state.hasError) {
      return (
        <MessageState
          title="This page hit a snag"
          action={(
            <button type="button" className="button" onClick={() => window.location.reload()}>
              Reload
            </button>
          )}
        >
          Reload and your practice history will still be here.
        </MessageState>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
