// Copyright 2022-2023, Collabora, Ltd.
//
// SPDX-License-Identifier: BSL-1.0
//
// Author: Ryan Pavlik <ryan.pavlik@collabora.com>

use std::{
    borrow::Borrow,
    collections::{hash_map, HashMap},
    fmt::Debug,
    hash::Hash,
};
use typed_index_collections::TiVec;

/// Indicates that your transform function for AtomTable did not return unique outputs,
/// so the "lookup ID by value" feature of the data structure could not be maintained in the
/// transformed result.
#[derive(Debug, thiserror::Error, Clone, Copy)]
#[error("The transform function output the same value for two unique input values, so an atom table cannot be constructed from the outputs")]
pub struct NonUniqueTransformOutputError;

/// Wraps a user-provided error type to also allow returning NonUniqueTransformOutputError
#[derive(Debug, thiserror::Error)]
pub enum TransformResError<E> {
    #[error("The transform function returned an error: {0}")]
    TransformFunctionError(#[source] E),
    #[error(transparent)]
    NonUniqueOutput(#[from] NonUniqueTransformOutputError),
}

impl<E> TransformResError<E> {
    pub fn as_non_unique_output(&self) -> Option<NonUniqueTransformOutputError> {
        if let Self::NonUniqueOutput(v) = self {
            Some(*v)
        } else {
            None
        }
    }
}

/// A data structure that lets you use strongly typed indices/IDs instead
/// of bulky values, performing a lookup in both directions.
///
/// T is your value type, and I is your index/ID type.
#[derive(Debug, Clone)]
pub struct AtomTable<T, I>
where
    I: From<usize>,
{
    vec: TiVec<I, T>,
    map: HashMap<T, I>,
}

impl<T, I> Default for AtomTable<T, I>
where
    I: From<usize> + Copy,
{
    fn default() -> Self {
        Self {
            vec: Default::default(),
            map: Default::default(),
        }
    }
}

impl<V, I> AtomTable<V, I>
where
    I: From<usize>,
{
    /// Look up the ID and return the corresponding value, if any.
    pub fn get_value(&self, id: I) -> Option<&V>
    where
        usize: From<I>,
    {
        self.vec.get(id)
    }

    /// Look up the provided, owned value, and either return the existing ID for it,
    /// or add it to the table and return the newly created ID for it.
    pub fn get_or_create_id_for_owned_value(&mut self, value: V) -> I
    where
        V: Clone + Hash + Eq,
        I: Copy,
    {
        if let Some(id) = self.map.get(&value) {
            return *id;
        }
        let id = self.vec.push_and_get_key(value.clone());
        self.map.insert(value, id);
        id
    }

    /// Look up the provided value, and either return the existing ID for it,
    /// or add it to the table and return the newly created ID for it.
    pub fn get_or_create_id(&mut self, value: &V) -> I
    where
        V: Clone + Hash + Eq,
        I: Copy,
    {
        if let Some(id) = self.map.get(value) {
            return *id;
        }
        let id = self.vec.push_and_get_key(value.clone());
        self.map.insert(value.clone(), id);
        id
    }

    /// Look up the provided value and return the existing ID for it, if any.
    pub fn get_id<Q: ?Sized>(&self, value: &Q) -> Option<I>
    where
        I: Copy,
        V: Hash + Eq + Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map.get(value).copied()
    }

    /// Apply a function to all values to produce a new atom table
    /// with correspondence between the IDs
    pub fn try_transform<U: Hash + Eq + Clone>(
        &self,
        mut f: impl FnMut(&V) -> U,
    ) -> Result<AtomTable<U, I>, NonUniqueTransformOutputError>
    where
        I: Eq + Debug,
    {
        self.try_transform_res(move |val| -> Result<U, ()> { Ok(f(val)) })
            .map_err(|e| {
                e.as_non_unique_output()
                    .expect("Nowhere to introduce a different kind of error")
            })
    }

    /// Apply a function returning Result<_, _> to all values to produce a new atom table
    /// with correspondence between the IDs.
    pub fn try_transform_res<U: Hash + Eq + Clone, E>(
        &self,
        mut f: impl FnMut(&V) -> Result<U, E>,
    ) -> Result<AtomTable<U, I>, TransformResError<E>>
    where
        I: Eq + Debug,
    {
        let mut vec: TiVec<I, U> = Default::default();
        let mut map: HashMap<U, I> = Default::default();
        vec.reserve(self.vec.len());
        for (id, val) in self.vec.iter_enumerated() {
            let new_val = f(val).map_err(TransformResError::TransformFunctionError)?;
            let new_id = vec.push_and_get_key(new_val.clone());
            assert_eq!(new_id, id);
            let old_id_for_this_val = map.insert(new_val, new_id);
            // Must not map more than one input to a single output
            if old_id_for_this_val.is_some() {
                return Err(TransformResError::from(NonUniqueTransformOutputError));
            }
            assert!(old_id_for_this_val.is_none());
        }
        Ok(AtomTable { vec, map })
    }
}

impl<V, I> Extend<V> for AtomTable<V, I>
where
    V: Hash + Eq + Clone,
    I: From<usize> + Copy,
{
    fn extend<T: IntoIterator<Item = V>>(&mut self, iter: T) {
        for val in iter {
            self.get_or_create_id_for_owned_value(val);
        }
    }
}

impl<V, I> FromIterator<V> for AtomTable<V, I>
where
    I: From<usize>,
    V: Clone + Hash + Eq,
{
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        let mut map = HashMap::new();
        let mut vec = TiVec::new();
        for val in iter {
            if let hash_map::Entry::Vacant(entry) = map.entry(val.clone()) {
                let id = vec.push_and_get_key(val);
                entry.insert(id);
            }
        }
        Self { vec, map }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Id(usize);

    impl From<usize> for Id {
        fn from(value: usize) -> Self {
            Self(value)
        }
    }
    impl From<Id> for usize {
        fn from(value: Id) -> Self {
            value.0
        }
    }

    #[test]
    fn read_basics() {
        let table: AtomTable<String, Id> = vec!["a", "b", "c"]
            .into_iter()
            .map(str::to_string)
            .collect();

        assert!(table.get_id("a").is_some());
        let a = table.get_id("a").unwrap();
        assert_eq!(table.get_value(a).unwrap(), "a");

        assert!(table.get_id("b").is_some());
        let b = table.get_id("b").unwrap();
        assert_eq!(table.get_value(b).unwrap(), "b");

        assert_ne!(a, b);

        assert!(table.get_id("c").is_some());
        let c = table.get_id("c").unwrap();
        assert_eq!(table.get_value(c).unwrap(), "c");

        assert_ne!(a, c);
        assert_ne!(b, c);

        assert!(table.get_id("d").is_none());

        // Construct an ID we expect to be bad
        let bad_id = Id::from(usize::from(c) + 200);

        assert_ne!(a, bad_id);
        assert_ne!(b, bad_id);
        assert_ne!(c, bad_id);
        assert!(table.get_value(bad_id).is_none());
    }

    #[test]
    fn mut_basics() {
        let mut table: AtomTable<String, Id> = vec!["a", "b", "c"]
            .into_iter()
            .map(str::to_string)
            .collect();

        let string_a = "a".to_string();
        let a = table.get_id(&string_a).unwrap();
        assert_eq!(table.get_or_create_id(&string_a), a);
        assert_eq!(table.get_or_create_id_for_owned_value(string_a.clone()), a);

        let string_b = "b".to_string();
        let b = table.get_id(&string_b).unwrap();
        assert_eq!(table.get_or_create_id(&string_b), b);
        assert_eq!(table.get_or_create_id_for_owned_value(string_b.clone()), b);

        let string_c = "c".to_string();
        let c = table.get_id(&string_c).unwrap();
        assert_eq!(table.get_or_create_id(&string_c), c);
        assert_eq!(table.get_or_create_id_for_owned_value(string_c.clone()), c);

        let string_d = "d".to_string();
        let d = table.get_or_create_id(&string_d);
        assert_ne!(a, d);
        assert_ne!(b, d);
        assert_ne!(c, d);
        assert!(table.get_id("d").is_some());
    }
}
