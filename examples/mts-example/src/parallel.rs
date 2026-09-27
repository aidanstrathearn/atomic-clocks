use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Folds indexed items into a bounded number of parallel partial results, then
/// merges those partials in item order.
///
/// `max_tasks` is treated as at least one and capped by the live Rayon pool and
/// item count. `start` creates each chunk's accumulator from its first item,
/// avoiding any requirement for an identity value. Successful results have a
/// stable reduction order for a given item count and task limit.
pub(super) fn try_fold_chunks_ordered<I, A, E, Start, Fold, Merge>(
    items: I,
    max_tasks: usize,
    start: Start,
    fold: Fold,
    merge: Merge,
) -> Result<Option<A>, E>
where
    I: IntoParallelIterator,
    I::Iter: IndexedParallelIterator,
    A: Send,
    E: Send,
    Start: Fn(I::Item) -> Result<A, E> + Sync + Send,
    Fold: Fn(&mut A, I::Item) -> Result<(), E> + Sync + Send,
    Merge: Fn(&mut A, A) -> Result<(), E> + Sync + Send,
{
    let items = items.into_par_iter();
    let item_count = items.len();
    if item_count == 0 {
        return Ok(None);
    }

    let task_count = max_tasks
        .max(1)
        .min(rayon::current_num_threads())
        .min(item_count);
    let chunk_size = item_count.div_ceil(task_count);
    let partials: Result<Vec<_>, E> = items
        .fold_chunks(
            chunk_size,
            || Ok::<_, E>(None),
            |partial, item| {
                let mut partial = partial?;
                if let Some(accumulator) = partial.as_mut() {
                    fold(accumulator, item)?;
                } else {
                    partial = Some(start(item)?);
                }
                Ok(partial)
            },
        )
        .collect();

    let mut partials = partials?.into_iter().flatten();
    let Some(mut result) = partials.next() else {
        return Ok(None);
    };
    for partial in partials {
        merge(&mut result, partial)?;
    }
    Ok(Some(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_none_without_starting_an_accumulator() {
        let result: Result<Option<String>, &'static str> = try_fold_chunks_ordered(
            0..0,
            4,
            |_| panic!("empty input must not start an accumulator"),
            |_, _| panic!("empty input must not fold an item"),
            |_, _| panic!("empty input must not merge partials"),
        );
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn preserves_item_order_across_chunks() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let result: Result<Option<String>, &'static str> = pool.install(|| {
            try_fold_chunks_ordered(
                0..7,
                3,
                |item| Ok(item.to_string()),
                |text, item| {
                    text.push_str(&item.to_string());
                    Ok(())
                },
                |text, partial| {
                    text.push_str(&partial);
                    Ok(())
                },
            )
        });
        assert_eq!(result.unwrap().unwrap(), "0123456");
    }

    #[test]
    fn propagates_start_fold_and_merge_errors() {
        let start: Result<Option<usize>, _> =
            try_fold_chunks_ordered(0..1, 1, |_| Err("start"), |_, _| Ok(()), |_, _| Ok(()));
        assert_eq!(start, Err("start"));

        let fold: Result<Option<usize>, _> =
            try_fold_chunks_ordered(0..2, 1, Ok, |_, _| Err("fold"), |_, _| Ok(()));
        assert_eq!(fold, Err("fold"));

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let merge: Result<Option<usize>, _> = pool
            .install(|| try_fold_chunks_ordered(0..2, 2, Ok, |_, _| Ok(()), |_, _| Err("merge")));
        assert_eq!(merge, Err("merge"));
    }
}
