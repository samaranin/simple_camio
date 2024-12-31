from abc import ABC
from typing import Dict, Optional, Type, TypeVar, cast

T = TypeVar("T", bound="Module")


class ModulesRepository:
    """
    Repository for accessing all the modules in the application.
    This is a singleton class and should be used to access all the modules.
    All modules inherit from the `Module` class and are automatically registered in the repository when first created.
    The repository can be accessed from any module by using the `repository` attribute.
    """

    __instance: Optional["ModulesRepository"] = None

    def __new__(cls) -> "ModulesRepository":
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self) -> None:
        self.__services: Dict[Type["Module"], "Module"] = dict()
        """ Dictionary of all registered modules. """

    def register(self, instance: "Module") -> None:
        """
        Register a module in the repository.
        If the module is already registered, it is replaced.
        """
        self.__services[type(instance)] = instance

    def get(self, cls: Type[T]) -> T:
        """
        Get a module from the repository by its class.
        If the module is not registered, a `ValueError` is raised.
        """
        service = self.__services.get(cls)
        if service is None:
            raise ValueError(f"Module {cls} not registered.")
        return cast(T, service)

    def __getitem__(self, cls: Type[T]) -> T:
        """
        Get a module from the repository by its class.
        """
        return self.get(cls)

    def __setitem__(self, cls: Type[T], instance: T) -> None:
        """
        Register a module in the repository.
        """
        self.register(instance)

    def __contains__(self, cls: Type[T]) -> bool:
        """
        Check if a module is registered in the repository.
        """
        return cls in self.__services


# Initialize the repository
_repository = ModulesRepository()


class Module(ABC):
    """
    Base class for all modules.
    """

    def __init__(self) -> None:
        self._repository = _repository
        """ Reference to the repository. """

        self._repository.register(self)


__all__ = ["ModulesRepository", "Module"]
