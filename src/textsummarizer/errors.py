class UserFacingError(Exception):
    """A failure the user can act on; the CLI reports it as one line, without a traceback."""
