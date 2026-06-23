'''Custom data containers with useful properties'''


from typing import (
    Callable,
    Collection,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    overload,
    Protocol,
    runtime_checkable,
    Sequence,
    TypeVar,
)
from collections import Counter, UserDict, defaultdict
from copy import deepcopy


T = TypeVar('T')
LabelT = TypeVar('LabelT', bound=Hashable)
HandleT = tuple[LabelT, int] # label uniquified with an additional arbitrary index

@runtime_checkable
class Labelled(Protocol):
    '''Protocol for objects that have a label'''
    @property
    def label(self) -> Hashable: 
        ...

class UniqueRegistry(UserDict, Generic[LabelT, T]):
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

    # Object registration
    def __setitem__(self, key : LabelT, item : T) -> None:
        raise PermissionError(f"Direct key-value assignment is not allowed; call 'register({item})' method instead")
    
    def _setitem(self, key : LabelT, item : T) -> None:
        '''Privatized version of __setitem__ - intend for internal use when copying UniqueRegistry objects'''
        super().__setitem__(key, item)

    @overload
    def register(self, obj : Labelled) -> HandleT:
        ...

    @overload
    def register(self, obj : T, label : LabelT) -> HandleT:
        ...
    
    @overload
    def register(self, obj : T, label : Callable[[T], LabelT]) -> HandleT:
        ...

    def register(self, obj : T | Labelled, label : Optional[Callable[[T], LabelT] | LabelT]=None) -> HandleT:
        '''Generate a new, unique handle for the given object and register it, then return the handle'''
        if label is None:
            if isinstance(obj, Labelled):
                label = obj.label
            else:
                raise TypeError(f'Cannot infer label from unlabelled object {obj!r}')
        elif isinstance(label, Callable): # N.B.: all Callables are Hashable, so order matters in any isinstance checks for the latter
            label = label(obj)

        handle : HandleT = (label, self._take_connector_number(label))
        super().__setitem__(handle, obj)

        return handle
    
    ## Composite registration methods
    def register_from_mapping(
        self,
        collection : Mapping[LabelT, Iterable[T]],
        # DEV: need to bundle Ts as iterable to allow passing mutiple objects w/
        # same label part of handle (mapping would be non-injective otherwise)
    ) -> list[HandleT]:
        '''
        Register objects from a mapping which maps labels to collections of objects
        
        All objects under the same label will have matching label parts but
        distinct IDs within their assigned handles post-registration

        For example, onto a previously-empty UniqueRegistry:
        >>> reg = UniqueRegistry()
        >>> reg.register_from_mapping({'foo' : (f1, f2), 'bar' : b1})
        would produce a registry like:
        {
            ('foo', 0) : f1,
            ('foo', 1) : f2,
            ('bar', 0) : b1,
        }
        '''
        handles : list[HandleT] = []
        for label, objs in collection.items():
            for obj in objs:
                handles.append(self.register(obj, label=label))
        return handles
    
    def register_from_sequential(
        self,
        collection : Sequence[T],
        label : Optional[Callable[[T], LabelT] | LabelT]=None,
    ) -> list[HandleT]:
        '''
        Register all objects from an iterable collection,
        with labels assigned according to a labeller rule which acts on those objects or,
        if no rule is provided BUT the objects are Labelled, the label attribute on those objects
        '''
        handles : list[HandleT] = []
        for obj in collection:
            handles.append(self.register(obj, label=label))
        return handles

    @overload
    def register_from(
        self,
        collection : Iterable[Labelled],
    ) -> list[HandleT]:
        ...

    @overload
    def register_from(
        self,
        collection : Iterable[T], # DEV: generic here is NOT a Labelled interface
        labeller : Callable[[T], LabelT],
    ) -> list[HandleT]:
        ...

    @overload
    def register_from(
        self,
        collection : Mapping[LabelT, Iterable[T]],
    ) -> list[HandleT]:
        ...

    def register_from(
        self,
        collection : Iterable[T],
        label : Optional[Callable[[T], LabelT] | LabelT]=None
    ) -> list[HandleT]:
        '''Register multiple objects at once, returning a list of their assigned handles'''
        if isinstance(collection, Mapping):
            if label is not None:
                raise ValueError('Registration from mapping received unexpected "labeller" argument')
            return self.register_from_mapping(collection)
        elif isinstance(collection, Iterable):
            return self.register_from_sequential(collection, label=label)

    # Object deregistration
    def deregister(self, handle : HandleT) -> T:
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
            
    # Read access
    @property
    def by_labels(self) -> dict[LabelT, tuple[T, ...]]: 
        # DEV: eventually would like to make sets (since order is irrelevant), but that relies on assumptions about hashability of T
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
          
    # Partitioning
    def split(self, partition : Collection[Iterable[HandleT]]) -> tuple['UniqueRegistry']:
        '''
        Separate this registry by a partition of its handle set
        
        Any handles not specifically explicitly identified in a part of the partition
        will be placed into a final, "implicit" registry, returned at the end
        '''
        ...

    def merge(self, other : 'UniqueRegistry', preserve_idxs : bool=False) -> None:
        '''
        Merge another registry into this one, with the handles of
        this registry taking priority in the case of collisions
        '''
        ...

    @classmethod
    def merged(cls, *other_regs : 'UniqueRegistry', preserve_idxs : bool=False) -> 'UniqueRegistry':
        '''
        Generate a registry by combining together other registries 
        Will preserve handle indices as-encountered when possible, even if they are non-contiguous

        Priority is first-come-first served, i.e. key collisions are
        resolved by giving the key to the object in the first seen registry
        '''
        reg = cls()
        for other_reg in other_regs:
            reg.merge(other_reg, preserve_idxs=preserve_idxs)
        
        return reg

    # Copying
    def copy(self, value_copy_method : Callable[[T], T]=deepcopy) -> 'UniqueRegistry[HandleT, T]':
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