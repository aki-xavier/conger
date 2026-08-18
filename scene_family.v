module conger

// scene_family.v — the common scene-family surface shared by the base and
// dynamic (child) codebooks, so InverseApp / ExpertRegistry can treat them
// uniformly (V's replacement for the Python Codebook class hierarchy).
import cga
import mlx

interface SceneFamily {
	to_scene(params []f64) cga.Scene
	sample(replicates int, seed u64, extrap bool) mlx.Array
	n_combo() int
	template_variant() string
	geometry_family() string
	template_lineage() TemplateLineage
}
