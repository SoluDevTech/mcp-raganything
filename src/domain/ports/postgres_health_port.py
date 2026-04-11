from abc import ABC, abstractmethod


class PostgresHealthPort(ABC):
    """Abstract port for PostgreSQL connectivity checks."""

    @abstractmethod
    async def ping(self) -> bool:
        """Check if PostgreSQL is reachable.

        Returns:
            True if the database connection succeeds, False otherwise.
        """
        pass
