I prefer **option (1)** too. A species should have one place where the user can inspect and edit its name, physical parameters, and equations. Renaming it should carry its equations along automatically.

My earlier explanation was incomplete. I described the database relationship without explaining why the application still breaks.

Today, a height-regression row points to a particular species record—say, species 17. Renaming species 17 from “Abete” to “Abete bianco” preserves that reference. But the application then turns those records into a cached table containing **names**, and the marks form searches that table by name. The species rename does not refresh the hypsometric table, so the form sees “Abete bianco” on one side and “Abete” on the other. That explains the failed height calculation in your experiment.

Volume has a separate problem: its equation lookup directly uses the species name. Refreshing the hypsometric table would therefore restore height calculation, but volume would still fail.

The intended behavior should be: **rename once, see the new name everywhere, retain every equation association.** Stable references, correct cache updates, and the unified settings UI all contribute to that.

For the unified UI, I would keep one row per species, with your two additional summary columns:

| Specie | Equazioni altezza | Equazione volume |
|---|---|---|
| Abete bianco | ✓ · 2 comprese | ✓ |
| Abete rosso | ✓ · 1 compresa | — |

These are illustrative counts. A height check mark should mean “equations exist”; the number makes partial regional coverage visible.

The species popup would contain three groups:

- **Species parameters:** common and Latin names, density, Pressler coefficient, classification, and existing flags.
- **Height equations:** one row per compresa, showing the formula and coefficients, with missing equations clearly indicated. Computed equations should also retain their sample count, fit quality, and source.
- **Volume equation:** formula, coefficients, units, and source. The user can inspect the available Tabacchi equations and select the association. Adding or editing an equation should be explicit and audited.

For Altra conifera and Altra latifoglia, the popup should clearly show that they use the Abete bianco and Faggio volume equations respectively. Sharing a volume equation must not automatically share a height regression.

I would unify this **in the interface**, while retaining separate database records for species, volume equations, and regional height equations. That accommodates any number of comprese without adding repeated columns or duplicating species parameters. Every equation association would use a stable reference; its displayed species name would always come from the species record.

Your batch-import proposal fits this well. I would provide separate height- and volume-import actions, both following the same process:

1. **Read the CSV without changing anything.** Continue accepting human-readable species and compresa names.
2. **Show a preview:** incoming name, matched species, compresa where applicable, existing coefficients, proposed coefficients, and any problems.
3. **Resolve unmatched names explicitly.** For example, an incoming “Abete” could be mapped to the existing “Abete bianco” through a selector. Ambiguous matches require a choice; the importer should not silently create another species or guess.
4. **Confirm or cancel.** Confirmation applies the reviewed changes together and records them in Controllo. Cancellation leaves everything untouched. If relevant settings changed while the preview was open, require a refreshed preview.

By default, I would update only the species/compresa combinations supplied in the file, preserving everything else. Replacing an entire equation collection could remain an explicit alternative, with the preview listing equations that would disappear. That is a deliberate change from the current whole-set height import behavior.

The same preservation rule matters when editing one species in its popup: changing its Capistrano regression must leave its other regional equations—and every other species’ equations—intact. We should retain the existing archive of previous height-parameter sets.

Finally, the Abete rosso example shows why equation availability needs to be visible during tree entry too. Missing volume equations should produce an explicit **“volume unavailable”** indication. Zero remains a valid calculated result for sufficiently small trees.

Adding the missing equation and repairing existing records should be separate operations. A repair preview could list affected trees, proposed volumes/masses, and intervention-total changes before anything is applied; it should not indiscriminately recompute every stored zero.

This makes option (1) a coherent workflow: one species editor, visible equation coverage, safe renames, and reviewed batch updates.
