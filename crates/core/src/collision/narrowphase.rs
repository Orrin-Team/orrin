//! Narrowphase: exact overlap tests per shape pair, producing a [`Contact`],
//! plus the MTV split used to resolve solid–solid penetration.
//!
//! Every function here upholds the module's normal convention: the returned
//! contact normal is unit length and points from the *first* argument's shape
//! toward the second's.

use glam::Vec3;

use super::{Aabb, Contact, WorldShape};

/// Dispatch to the right shape–shape test. The Sphere/Box case reuses the
/// Box/Sphere test with the normal flipped to keep the a→b convention.
pub fn test(a: &WorldShape, b: &WorldShape) -> Option<Contact> {
    match (a, b) {
        (WorldShape::Box(a), WorldShape::Box(b)) => aabb_aabb(a, b),
        (
            WorldShape::Sphere { center: ca, radius: ra },
            WorldShape::Sphere { center: cb, radius: rb },
        ) => sphere_sphere(*ca, *ra, *cb, *rb),
        (WorldShape::Box(a), WorldShape::Sphere { center, radius }) => {
            aabb_sphere(a, *center, *radius)
        }
        (WorldShape::Sphere { center, radius }, WorldShape::Box(b)) => {
            aabb_sphere(b, *center, *radius)
                .map(|contact| Contact { normal: -contact.normal, ..contact })
        }
    }
}

fn aabb_aabb(a: &Aabb, b: &Aabb) -> Option<Contact> {
    let extent = a.max.min(b.max) - a.min.max(b.min);
    let delta = b.center() - a.center();

    if extent.x <= 0.0 || extent.y <= 0.0 || extent.z <= 0.0 { return None }

    // The smallest-overlap axis is what makes the MTV minimum: pushing out
    // along any other axis moves further. `delta` only decides the sign.
    let (depth, normal) = if extent.x <= extent.y && extent.x <= extent.z {
        (
            extent.x,
            if delta.x >= 0.0 { Vec3::X } else { Vec3::NEG_X }
        )
    }
    else if extent.y <= extent.z {
        (
            extent.y,
            if delta.y >= 0.0 { Vec3::Y } else { Vec3::NEG_Y }
        )
    }
    else {
        (
            extent.z,
            if delta.z >= 0.0 { Vec3::Z } else { Vec3::NEG_Z }
        )
    };

    let overlap_min = a.min.max(b.min);
    let overlap_max = a.max.min(b.max);

    let point = (overlap_min + overlap_max) * 0.5;

    Some(Contact {
        normal,
        point,
        depth
    })
}

fn sphere_sphere(ca: Vec3, ra: f32, cb: Vec3, rb: f32) -> Option<Contact> {
    let d = cb - ca;
    let dist = d.length();
    if dist < ra + rb {
        // Concentric spheres have no separation direction; any unit vector
        // is as good as any other, and NaN is not.
        let normal = d.normalize_or(Vec3::X);

        return Some(Contact {
            normal,
            point: (ca + normal * ra),
            depth: (ra + rb) - dist,
        })
    }
    None
}

fn aabb_sphere(a: &Aabb, center: Vec3, radius: f32) -> Option<Contact> {
    let closest = center.clamp(a.min, a.max);
    let d = center - closest;
    let dist = d.length();

    if dist >= radius { return None }

    // Center inside the box: `d` is zero (exact — clamp returns the center
    // unchanged), so there's no direction to normalize. Push out through the
    // nearest face instead; depth must clear that face plus the radius.
    if dist == 0.0 {
        let to_min = center - a.min;
        let to_max = a.max - center;

        let candidates = [
            (to_min.x, Vec3::NEG_X), (to_max.x, Vec3::X),
            (to_min.y, Vec3::NEG_Y), (to_max.y, Vec3::Y),
            (to_min.z, Vec3::NEG_Z), (to_max.z, Vec3::Z),
        ];

        let (face_dist, normal) = candidates.iter().min_by(|x, y| x.0.total_cmp(&y.0)).unwrap();

        return Some(Contact {
            normal: *normal,
            depth: face_dist + radius,
            point: center
        })
    }

    Some(Contact {
        normal: d / dist,
        depth: radius - dist,
        point: closest
    })
}

/// Split a solid–solid penetration into per-body displacement offsets:
/// returns `(offset_a, offset_b)` for the contact's `a` and `b` entities.
pub fn resolve_offsets(contact: &Contact) -> (Vec3, Vec3) {
    let mtv = contact.normal * contact.depth;
    (-mtv * 0.5, mtv * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box_at(center: Vec3) -> Aabb {
        Aabb { min: center - Vec3::splat(0.5), max: center + Vec3::splat(0.5) }
    }

    #[test]
    fn separated_shapes_do_not_collide() {
        assert!(aabb_aabb(&unit_box_at(Vec3::ZERO), &unit_box_at(Vec3::new(3.0, 0.0, 0.0))).is_none());
        assert!(sphere_sphere(Vec3::ZERO, 0.5, Vec3::new(3.0, 0.0, 0.0), 0.5).is_none());
        assert!(aabb_sphere(&unit_box_at(Vec3::ZERO), Vec3::new(3.0, 0.0, 0.0), 0.5).is_none());
    }

    #[test]
    fn aabb_aabb_picks_smallest_axis() {
        // Offset 0.8 on x, 0.2 on y: overlap is 0.2 on x, 0.8 on y, 1.0 on z,
        // so the MTV must be along +x with depth 0.2.
        let a = unit_box_at(Vec3::ZERO);
        let b = unit_box_at(Vec3::new(0.8, 0.2, 0.0));
        let contact = aabb_aabb(&a, &b).expect("boxes overlap");
        assert!((contact.normal - Vec3::X).length() < 1e-5);
        assert!((contact.depth - 0.2).abs() < 1e-5);
    }

    #[test]
    fn sphere_sphere_reports_depth_and_direction() {
        let contact =
            sphere_sphere(Vec3::ZERO, 0.5, Vec3::new(0.8, 0.0, 0.0), 0.5).expect("spheres overlap");
        assert!((contact.normal - Vec3::X).length() < 1e-5);
        assert!((contact.depth - 0.2).abs() < 1e-5);
    }

    #[test]
    fn aabb_sphere_from_the_side() {
        // Sphere just left of the box's -x face, overlapping by 0.2.
        let contact = aabb_sphere(&unit_box_at(Vec3::ZERO), Vec3::new(-0.8, 0.0, 0.0), 0.5)
            .expect("overlapping");
        // Normal points box → sphere, i.e. -x.
        assert!((contact.normal - Vec3::NEG_X).length() < 1e-5);
        assert!((contact.depth - 0.2).abs() < 1e-5);
    }

    #[test]
    fn aabb_sphere_center_inside_pushes_through_nearest_face() {
        // Sphere center inside the box, nearest the +x face (0.1 away): the
        // fallback must push +x, and depth must clear face distance + radius.
        let contact = aabb_sphere(&unit_box_at(Vec3::ZERO), Vec3::new(0.4, 0.0, 0.0), 0.5)
            .expect("center inside the box always overlaps");
        assert!((contact.normal - Vec3::X).length() < 1e-5);
        assert!((contact.depth - 0.6).abs() < 1e-5);
    }

    #[test]
    fn resolve_offsets_split_the_mtv() {
        let contact = Contact { point: Vec3::ZERO, normal: Vec3::X, depth: 0.2 };
        let (offset_a, offset_b) = resolve_offsets(&contact);
        assert!((offset_a - Vec3::new(-0.1, 0.0, 0.0)).length() < 1e-5);
        assert!((offset_b - Vec3::new(0.1, 0.0, 0.0)).length() < 1e-5);
        // Together they exactly cancel the penetration.
        assert!(((offset_b - offset_a).length() - contact.depth).abs() < 1e-5);
    }
}
