class Conflict(Exception):
    """Raised when a resource already exists and cannot be created again."""


class NotFound(Exception):
    """Raised when a resource is not found."""


class InvalidReference(Exception):
    """Raised when a referenced resource does not exist."""


class InUse(Exception):
    """Raised when a resource cannot be deleted while it is referenced."""
