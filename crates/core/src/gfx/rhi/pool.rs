//! Slot storage behind the RHI's typed handles.
//!
//! A handle is a dense `u32` index the driver owns. Destruction is a request
//! stamped with the frame it was made in; the slot is dropped, and only then
//! reusable, once `frames_in_flight` further frames have retired, because any of
//! them may still have recorded a command that names it. This is the one place
//! that rule is written down, in place of each pass holding its resources alive
//! by hand.

pub(crate) struct Pool<T> {
    slots: Vec<Option<T>>,
    pending: Vec<bool>,
    retiring: Vec<(u32, u64)>,
    free: Vec<u32>,
    frames_in_flight: u64,
}

impl<T> Pool<T> {
    pub(crate) fn new(frames_in_flight: u64) -> Self {
        Self {
            slots: Vec::new(),
            pending: Vec::new(),
            retiring: Vec::new(),
            free: Vec::new(),
            frames_in_flight,
        }
    }

    pub(crate) fn insert(&mut self, value: T) -> u32 {
        match self.free.pop() {
            Some(index) => {
                self.slots[index as usize] = Some(value);
                self.pending[index as usize] = false;
                index
            }
            None => {
                self.slots.push(Some(value));
                self.pending.push(false);
                (self.slots.len() - 1) as u32
            }
        }
    }

    pub(crate) fn get(&self, index: u32) -> Option<&T> {
        self.slots.get(index as usize)?.as_ref()
    }

    pub(crate) fn get_mut(&mut self, index: u32) -> Option<&mut T> {
        self.slots.get_mut(index as usize)?.as_mut()
    }

    /// Queue `index` for destruction as of `frame`. Freeing a slot twice is
    /// ignored: recycling it twice would hand one slot to two owners.
    pub(crate) fn remove(&mut self, index: u32, frame: u64) {
        let Some(pending) = self.pending.get_mut(index as usize) else {
            return;
        };
        if *pending || self.slots[index as usize].is_none() {
            return;
        }
        *pending = true;
        self.retiring.push((index, frame));
    }

    /// Drop every slot whose last possible use has retired by `frame`.
    pub(crate) fn collect(&mut self, frame: u64) {
        let depth = self.frames_in_flight;
        let (slots, free) = (&mut self.slots, &mut self.free);
        self.retiring.retain(|&(index, removed)| {
            let retired = removed + depth <= frame;
            if retired {
                slots[index as usize] = None;
                free.push(index);
            }
            !retired
        });
    }

    #[cfg(test)]
    pub(crate) fn live(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_is_not_reused_until_every_frame_that_could_use_it_has_retired() {
        let mut pool = Pool::<&str>::new(2);
        let a = pool.insert("a");
        pool.remove(a, 10);
        assert_eq!(pool.get(a), Some(&"a"), "removal only queues the free");

        pool.collect(10);
        pool.collect(11);
        assert_eq!(pool.get(a), Some(&"a"), "frame 10 may still be on the GPU");
        assert_eq!(pool.insert("b"), a + 1, "so its slot is not handed out");

        pool.collect(12);
        assert_eq!(pool.get(a), None);
        assert_eq!(pool.insert("c"), a, "once retired the slot is recycled");
    }

    #[test]
    fn handles_are_dense_indices_from_zero() {
        let mut pool = Pool::<u8>::new(1);
        assert_eq!([pool.insert(1), pool.insert(2), pool.insert(3)], [0, 1, 2]);
        assert_eq!(pool.live(), 3);
    }

    #[test]
    fn a_double_free_is_ignored_rather_than_recycling_a_slot_twice() {
        let mut pool = Pool::<u8>::new(1);
        let a = pool.insert(1);
        pool.remove(a, 0);
        pool.remove(a, 0);
        pool.collect(1);
        let (x, y) = (pool.insert(2), pool.insert(3));
        assert_ne!(x, y);
    }
}
