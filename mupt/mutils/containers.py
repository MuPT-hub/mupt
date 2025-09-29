'''Custom data containers with useful properties'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import (
    Callable,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
)
from collections import Counter, UserDict, defaultdict
from copy import deepcopy


LabelT = TypeVar('LabelT', bound=Hashable)
HandleT = tuple[LabelT, int] # label uniquified with an additional arbitrary index

class Labelled(Protocol):
    '''Protocol for objects that have a label'''
    @property
    def label(self) -> Hashable: 
        ...
LabelledT = TypeVar('LabelledT', bound=Labelled)

class UniqueRegistry(UserDict, Generic[LabelT, LabelledT]):
    '''
    A registry of Labelled objects which are each assigned a unique "handle",
    comprising the object's label and a unique integer index determined by its time of insertion
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)
        self._ticker = Counter()
        self._freed = defaultdict(set)

    # Unique index ticker management
    def _take_connector_number(self, label : LabelT) -> int:
        '''Increment and return the next available integer for uniquifying a Connector label'''
        if len(freed_idxs := self._freed[label]) > 0:
            idx = min(freed_idxs)
            self._freed[label].remove(idx)
        else:
            idx = self._ticker[label]
            self._ticker[label] += 1

        return idx
    
    def reset_ticker(self) -> None:
        '''Reset all unique index counters to zero'''
        self._ticker = Counter()
        
    def adjust_ticker_count_for(self, label : LabelT, n: int) -> None:
        '''Adjust the unique index counter for a given label to the given integer'''
        self._ticker[label] = n
        
    def reset_ticker_count_for(self, label : LabelT) -> None:
        '''Reset the unique index counter for a given label to zero'''
        self.adjust_ticker_count_for(label, 0)

    # Labelled object registration
    ## Object registration
    def __setitem__(self, key : LabelT, item : LabelledT) -> None:
        raise PermissionError(f"Direct key-value assignment is not allowed; call 'register({item})' method instead")
    
    def _setitem(self, key : LabelT, item : LabelledT) -> None:
        '''Privatized version of __setitem__ - intend for internal use when copying UniqueRegistry objects'''
        super().__setitem__(key, item)

    def register(self, obj: LabelledT, label : Optional[LabelT]=None) -> HandleT:
        '''Generate a new, unique handle for the given object and register it, then return the handle'''
        if label is None:
            label = obj.label # DEV: opted for behavioral pattern, rather than explicit runtime_checkable Protocol enforcement
        handle = (label, self._take_connector_number(label))
        super().__setitem__(handle, obj)

        return handle
    
    def register_from(self, collection : Iterable[LabelledT]) -> list[HandleT]:
        '''Register multiple objects at once, returning a list of their assigned handles'''
        handles : list[HandleT] = []
        if isinstance(collection, Mapping):
            for label, obj in collection.items():
                handles.append(self.register(obj, label=label))
        else:
            for obj in collection:
                handles.append(self.register(obj, label=None))

        return handles

    ## Object deregistration
    def deregister(self, handle : HandleT) -> LabelledT:
        '''
        Unregister the object with the given handle and free the index assigned to that object
        Returns the objects bound to that handle
        '''
        obj = self.pop(handle)
        label, idx = handle
        self._freed[label].add(idx)
        
        return obj

    def purge(self, label : LabelT) -> None:
        '''Unregister all objects with the given label'''
        handles_to_remove = [handle for handle in self.keys() if handle[0] == label]
        for handle in handles_to_remove:
            self.deregister(handle)
            
    ## Object access
    @property
    def by_labels(self) -> dict[LabelT, tuple[LabelledT, ...]]: 
        # DEV: eventually would like to make sets (since order is irrelevant), but that relies on assumptions about hashability of LabelledT
        '''
        Mapping from labels (without uniquifying handle index) to classes of objects registered to those labels
        Can be thought of as the equivalence classes of objects under the relation "o1.handle[0] == o2.handle[0]"
        '''
        label_classes = defaultdict(list)
        for (label, idx), child in self.items():
            label_classes[label].append(child)
            
        return { # downconvert from defaultdict -> dict and make values collections immutable by tuple-ifying them
            label : tuple(child_class)
                for label, child_class in label_classes.items()
        }
          
    # Copying
    def copy(self, value_copy_method : Callable[[LabelledT], LabelledT]=deepcopy) -> 'UniqueRegistry[HandleT, LabelledT]':
        '''
        Create a deep copy of this UniqueRegistry, with the same (key, value) pairs and internal state
        Requires a method for copying values in general, since their complete type is not explicit a priori
        '''
        new_registry = UniqueRegistry()
        new_registry._ticker = Counter(self._ticker)
        new_registry._freed = defaultdict(
            set,
            **{ # DEV: this looks elaborate, but is necessary to ensure copy doesn't share state with self after creation
                label : set(free_idxs)
                    for label, free_idxs in self._freed.items()
            }
        )
        for handle, obj in self.items():
            new_registry._setitem(handle, value_copy_method(obj))
            
        return new_registry