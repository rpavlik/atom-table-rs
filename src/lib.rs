// Copyright 2022-2023, Collabora, Ltd.
//
// SPDX-License-Identifier: BSL-1.0
//
// Author: Ryan Pavlik <ryan.pavlik@collabora.com>

//! This crate provides a generic "Atom Table" style data structure.
//! "Atom Table" conventionally refers to a table of strings where each has an ID,
//! and you can look up the ID by the string, or the string by the ID.
//! The [`AtomTable<V, I>`][`AtomTable`] struct in this crate is more generic,
//! allowing nearly any type as the "value" type, and using custom type-safe ID types
//! you provide as the ID.
//!
//! # Examples
//!
//! Example usage might look like:
//!
//! ```rust
//! use derive_more::{From, Into};
//! use atom_table::AtomTable;
//!
//! /// The ID type for the table
//! /// Using `derive_more` to save having to manually implement traits
//! #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, From, Into)]
//! struct Id(usize);
//!
//! let mut table: AtomTable<String, Id> = vec!["a", "b", "c"]
//!     .into_iter()
//!     .map(str::to_string)
//!     .collect();
//!
//! let a = table.get_id("a").unwrap();
//! let b = table.get_id("b").unwrap();
//! let d = table.get_or_create_id_for_owned_value("d".to_string());
//!
//! let b_try_again = table.get_or_create_id_for_owned_value("b".to_string());
//! assert_eq!(b, b_try_again);
//! ```
//!
#![warn(
    clippy::all,
    future_incompatible,
    missing_copy_implementations,
    missing_debug_implementations,
    missing_docs,
    nonstandard_style,
    rust_2018_idioms,
    rust_2021_compatibility,
    single_use_lifetimes,
    trivial_casts,
    unreachable_pub,
    unused
)]

use std::{
    borrow::Borrow,
    collections::{hash_map, HashMap},
    fmt::Debug,
    hash::Hash,
};
use typed_index_collections::TiVec;

/// A data structure that lets you use strongly typed indices/IDs instead
/// of bulky values, performing a lookup in both directions.
///
/// `T` is your value type, and `I` is your index/ID type.
///
/// Typically I should be a "newtype" (tuple struct) wrapping usize
/// and with [`Copy`] and [`From<usize>`] implemented for it and [`From<I>`] implemented for usize.
/// The `derive_more` crate may be useful for implementing these traits.
///
/// Right now, the values must have an implementation of [`Clone`]. This may be loosened
/// in the future if there is a straightforward and compelling way and reason to do so.
///
/// Values cannot be modified once inserted, because remaining the same is essential to providing
/// the invariant (and, each value is stored twice within the structure at this time)
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

impl<V, I> PartialEq for AtomTable<V, I>
where
    I: From<usize>,
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        // do not need to compare the map, it is basically just a cache
        self.vec == other.vec
    }
}

impl<V, I> Eq for AtomTable<V, I>
where
    I: From<usize>,
    V: Eq,
{
}

impl<V, I> AtomTable<V, I>
where
    I: From<usize>,
{
    /// Make a new (empty) table.
    pub fn new() -> Self
    where
        I: Copy,
    {
        Default::default()
    }

    /// Iterate over the values.
    pub fn iter(&self) -> impl Iterator<Item = &V> {
        self.vec.iter()
    }

    /// Iterate over the (ID, value) pairs.
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (I, &V)> {
        self.vec.iter_enumerated()
    }

    /// Look up the ID and return the corresponding value, if any.
    pub fn get(&self, id: I) -> Option<&V>
    where
        usize: From<I>,
    {
        self.vec.get(id)
    }

    /// Look up the provided, owned value, and either return the existing ID for it,
    /// or add it to the table and return the newly created ID for it.
    ///
    /// - See [`AtomTable::get_id`] if you want to only get the ID if one already exists.
    /// - See [`AtomTable::get_or_create_id`] if you do not already own the value or are unwilling
    ///   to transfer ownership - it will do an extra clone for you, as late as possible, if necessary.
    pub fn get_or_create_id_for_owned_value(&mut self, value: V) -> I
    where
        V: Clone + Hash + Eq,
        I: Copy,
    {
        if let Some(id) = self.map.get(&value) {
            return *id;
        }
        let id = self.vec.push_and_get_key(value.clone());
        let insert_result = self.map.insert(value, id);
        assert!(insert_result.is_none());
        id
    }

    /// Look up the provided value, and either return the existing ID for it,
    /// or add it to the table and return the newly created ID for it.
    ///
    /// - See [`AtomTable::get_id`] if you want to only get the ID if one already exists.
    /// - See [`AtomTable::get_or_create_id_for_owned_value`] if you already own the value and
    ///   are willing to transfer ownership: it will save one clone.
    pub fn get_or_create_id(&mut self, value: &V) -> I
    where
        V: Clone + Hash + Eq,
        I: Copy,
    {
        if let Some(id) = self.map.get(value) {
            return *id;
        }
        let id = self.vec.push_and_get_key(value.clone());
        let insert_result = self.map.insert(value.clone(), id);
        assert!(insert_result.is_none());
        id
    }

    /// Look up the provided value and return the existing ID for it, if any.
    ///
    /// The generic type usage here mirrors that used in [`HashMap<K, V>::get`], to allow e.g. `&str` to be
    /// passed here if the value type is [`String`].
    pub fn get_id<Q: ?Sized>(&self, value: &Q) -> Option<I>
    where
        I: Copy,
        V: Hash + Eq + Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map.get(value).copied()
    }
}

impl<V, I> IntoIterator for AtomTable<V, I>
where
    I: From<usize>,
{
    type Item = V;

    type IntoIter = <TiVec<I, V> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

impl<V, I> Extend<V> for AtomTable<V, I>
where
    V: Hash + Eq + Clone,
    I: From<usize> + Copy,
{
    fn extend<T: IntoIterator<Item = V>>(&mut self, iter: T) {
        for val in iter {
            let _ = self.get_or_create_id_for_owned_value(val);
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
                let _ = entry.insert(id);
            }
        }
        Self { vec, map }
    }
}

impl<V, I> From<AtomTable<V, I>> for Vec<V>
where
    I: From<usize>,
{
    fn from(value: AtomTable<V, I>) -> Self {
        value.vec.into()
    }
}

// Support for the "transform" functions/feature follows

/// Indicates that your transform function for AtomTable did not return unique outputs,
/// so the "lookup ID by value" feature of the data structure could not be maintained in the
/// transformed result.
#[cfg(feature = "transform")]
#[derive(Debug, thiserror::Error, Clone, Copy)]
#[error("The transform function output the same value for two unique input values, so an atom table cannot be constructed from the outputs")]
pub struct NonUniqueTransformOutputError;

/// Wraps a user-provided error type to also allow returning [`NonUniqueTransformOutputError`]
#[cfg(feature = "transform")]
#[derive(Debug, thiserror::Error)]
pub enum TransformResError<E> {
    /// Wraps a user-provided error type used by a transform function
    #[error("The transform function returned an error: {0}")]
    TransformFunctionError(#[source] E),
    /// Wraps [`NonUniqueTransformOutputError`]
    #[error(transparent)]
    NonUniqueOutput(#[from] NonUniqueTransformOutputError),
}

#[cfg(feature = "transform")]
impl<E> TransformResError<E> {
    fn as_non_unique_output(&self) -> Option<NonUniqueTransformOutputError> {
        if let Self::NonUniqueOutput(v) = self {
            Some(*v)
        } else {
            None
        }
    }
}

#[cfg(feature = "transform")]
impl<V, I> AtomTable<V, I>
where
    I: From<usize>,
{
    /// Apply a function to all values to produce a new atom table
    /// with correspondence between the IDs.
    ///
    /// Returns an [`NonUniqueTransformOutputError`] error if your function does not return
    /// unique outputs for each input, which would break the invariant of the data structure.
    ///
    /// Requires (default) feature "transform"
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

    /// Apply a function returning [`Result<T, E>`] to all values to produce a new atom table
    /// with correspondence between the IDs.
    /// (That is, `new_table.get(id)` on the new table will return effectively `old_table.get(id).map(p).ok().flatten()`)
    ///
    /// Requires (default) feature "transform"
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
        assert_eq!(table.get(a).unwrap(), "a");

        assert!(table.get_id("b").is_some());
        let b = table.get_id("b").unwrap();
        assert_eq!(table.get(b).unwrap(), "b");

        assert_ne!(a, b);

        assert!(table.get_id("c").is_some());
        let c = table.get_id("c").unwrap();
        assert_eq!(table.get(c).unwrap(), "c");

        assert_ne!(a, c);
        assert_ne!(b, c);

        assert!(table.get_id("d").is_none());

        // Construct an ID we expect to be bad
        let bad_id = Id::from(usize::from(c) + 200);

        assert_ne!(a, bad_id);
        assert_ne!(b, bad_id);
        assert_ne!(c, bad_id);
        assert!(table.get(bad_id).is_none());
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
        assert_eq!(table.get_id("d"), Some(d));

        table.extend(vec![
            "d".to_string(),
            "e".to_string(),
            "f".to_string(),
            "g".to_string(),
        ]);

        assert_eq!(table.get_id("d"), Some(d));
        assert!(table.get_id("e").is_some());
        assert!(table.get_id("f").is_some());
        assert!(table.get_id("g").is_some());
        let e = table.get_id("e").unwrap();
        let f = table.get_id("f").unwrap();
        let g = table.get_id("g").unwrap();

        let ids = vec![a, b, c, d, e, f, g];
        let unique_ids: HashSet<Id> = ids.iter().copied().collect();

        assert_eq!(ids.len(), unique_ids.len());
    }
}
