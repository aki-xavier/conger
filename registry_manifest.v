module conger

// registry_manifest.v — JSON persistence for dynamic child templates and the
// expert registry. Uses json2's struct
// codegen directly, so the typed ChildTemplateSpec / TemplateConstraints
// round-trip without a hand-written Any conversion.
import json2
import os

// RegisteredChildTemplate records a trained dynamic child template.
pub struct RegisteredChildTemplate {
pub:
	spec       ChildTemplateSpec
	model_path string
}

// RegistryManifest is the strong-typed registry_manifest.json.
pub struct RegistryManifest {
pub:
	children []RegisteredChildTemplate
	pending  []ChildTemplateSpec
	version  int = 1
}

// rm_save writes the manifest to path (creating parent directories).
pub fn rm_save(mf RegistryManifest, path string) ! {
	dir := os.dir(path)
	if dir != '.' && dir != '' && !os.exists(dir) {
		os.mkdir_all(dir, os.MkdirParams{})!
	}
	os.write_file(path, json2.encode(mf, json2.EncoderOptions{}))!
}

// rm_load reads and parses the manifest at path.
pub fn rm_load(path string) !RegistryManifest {
	content := os.read_file(path)!
	return json2.decode[RegistryManifest](content, json2.DecoderOptions{})!
}
