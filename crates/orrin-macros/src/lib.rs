//! `#[derive(Reflect)]` — generates a type's conversion to and from
//! `orrin_registry::Value`.
//!
//! Re-exported from `orrin-registry`; use it from there rather than depending
//! on this crate directly. Generated code is fully path-qualified, so no import
//! beyond the derive itself is needed at the use site.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Attribute, Data, DataEnum, DataStruct, DeriveInput, Fields, parse_macro_input};

/// Derive `Reflect` for a struct or enum whose fields are all `Reflect`.
///
/// Supported shapes: named structs, newtype structs (which flatten to their
/// inner value), unit structs, and enums with named-field or unit variants.
/// Tuple structs with more than one field and tuple variants are rejected —
/// positional names in a scene file are a migration hazard, and `Value` has no
/// representation for them.
///
/// `#[reflect(skip)]` on a field leaves it out of the value entirely; it is
/// restored with `Default::default()`, so a skipped field's type must be
/// `Default`. This is the Rust counterpart of C#'s `[Transient]`.
///
/// A type whose constructor establishes an invariant its fields do not must
/// *not* use this derive — the generated `from_value` assigns fields directly
/// and would happily build an instance the constructor would have rejected.
/// `Spin` is the engine's example.
#[proc_macro_derive(Reflect, attributes(reflect))]
pub fn derive_reflect(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let (to_body, from_body) = match &input.data {
        Data::Struct(data) => struct_bodies(data)?,
        Data::Enum(data) => enum_bodies(data)?,
        Data::Union(_) => {
            return Err(syn::Error::new_spanned(
                name,
                "Reflect cannot be derived for a union",
            ));
        }
    };

    Ok(quote! {
        impl #impl_generics ::orrin_registry::Reflect for #name #ty_generics #where_clause {
            fn to_value(&self) -> ::orrin_registry::Value {
                #to_body
            }

            fn from_value(
                value: &::orrin_registry::Value,
            ) -> ::core::result::Result<Self, ::orrin_registry::ValueError> {
                #from_body
            }
        }
    })
}

fn struct_bodies(data: &DataStruct) -> syn::Result<(TokenStream2, TokenStream2)> {
    match &data.fields {
        Fields::Named(fields) => {
            let mut entries = Vec::new();
            let mut inits = Vec::new();

            for field in &fields.named {
                let ident = field.ident.as_ref().expect("named field");
                let key = ident.to_string();
                if skipped(&field.attrs)? {
                    inits.push(quote! { #ident: ::core::default::Default::default() });
                } else {
                    entries.push(quote! {
                        (
                            ::std::string::String::from(#key),
                            ::orrin_registry::Reflect::to_value(&self.#ident),
                        )
                    });
                    inits.push(quote! { #ident: ::orrin_registry::take(value, #key)? });
                }
            }

            // Nothing to read means `value` goes untouched, and the generated
            // code would warn about it.
            let unused = entries.is_empty().then(|| quote! { let _ = value; });

            Ok((
                quote! { ::orrin_registry::Value::Struct(::std::vec![#(#entries),*]) },
                quote! {
                    #unused
                    ::core::result::Result::Ok(Self { #(#inits),* })
                },
            ))
        }

        // A newtype has no field name worth writing, so it flattens: `Name` is
        // a string in the file, not a struct wrapping one. Field paths gain no
        // meaningless `.0` level either.
        Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
            let field = &fields.unnamed[0];
            if skipped(&field.attrs)? {
                return Err(syn::Error::new_spanned(
                    field,
                    "`skip` on a newtype's only field would discard the whole value",
                ));
            }
            let ty = &field.ty;
            Ok((
                quote! { ::orrin_registry::Reflect::to_value(&self.0) },
                quote! {
                    <#ty as ::orrin_registry::Reflect>::from_value(value).map(Self)
                },
            ))
        }

        Fields::Unnamed(fields) => Err(syn::Error::new_spanned(
            fields,
            "Reflect supports newtype tuple structs only — give the fields names, \
             since positional keys in a scene file cannot survive a reordering",
        )),

        Fields::Unit => Ok((
            quote! { ::orrin_registry::Value::Struct(::std::vec::Vec::new()) },
            quote! {
                let _ = value;
                ::core::result::Result::Ok(Self)
            },
        )),
    }
}

fn enum_bodies(data: &DataEnum) -> syn::Result<(TokenStream2, TokenStream2)> {
    let mut to_arms = Vec::new();
    let mut from_arms = Vec::new();
    let mut names = Vec::new();

    for variant in &data.variants {
        let ident = &variant.ident;
        let key = ident.to_string();
        names.push(key.clone());

        match &variant.fields {
            Fields::Named(fields) => {
                let mut bindings = Vec::new();
                let mut entries = Vec::new();
                let mut inits = Vec::new();

                for field in &fields.named {
                    let field_ident = field.ident.as_ref().expect("named field");
                    let field_key = field_ident.to_string();
                    if skipped(&field.attrs)? {
                        bindings.push(quote! { #field_ident: _ });
                        inits.push(quote! { #field_ident: ::core::default::Default::default() });
                    } else {
                        bindings.push(quote! { #field_ident });
                        entries.push(quote! {
                            (
                                ::std::string::String::from(#field_key),
                                ::orrin_registry::Reflect::to_value(#field_ident),
                            )
                        });
                        inits.push(quote! {
                            #field_ident: ::orrin_registry::take(value, #field_key)?
                        });
                    }
                }

                to_arms.push(quote! {
                    Self::#ident { #(#bindings),* } => ::orrin_registry::Value::Enum {
                        variant: ::std::string::String::from(#key),
                        fields: ::std::vec![#(#entries),*],
                    }
                });
                from_arms.push(quote! {
                    #key => ::core::result::Result::Ok(Self::#ident { #(#inits),* })
                });
            }

            Fields::Unit => {
                to_arms.push(quote! {
                    Self::#ident => ::orrin_registry::Value::Enum {
                        variant: ::std::string::String::from(#key),
                        fields: ::std::vec::Vec::new(),
                    }
                });
                from_arms.push(quote! {
                    #key => ::core::result::Result::Ok(Self::#ident)
                });
            }

            Fields::Unnamed(fields) => {
                return Err(syn::Error::new_spanned(
                    fields,
                    "Reflect supports named-field and unit variants only — give the \
                     fields names, since positional keys in a scene file cannot \
                     survive a reordering",
                ));
            }
        }
    }

    // Listed in the error so a scene naming a renamed variant says what the
    // type does have. Built here because the variant set is known at expansion.
    let expected = format!("one of: {}", names.join(", "));

    Ok((
        quote! {
            match self {
                #(#to_arms),*
            }
        },
        quote! {
            let ::core::option::Option::Some(variant) = value.variant() else {
                return ::core::result::Result::Err(
                    ::orrin_registry::ValueError::mismatch("enum", value),
                );
            };
            match variant {
                #(#from_arms,)*
                other => ::core::result::Result::Err(
                    ::orrin_registry::ValueError::unknown_variant(#expected, other),
                ),
            }
        },
    ))
}

fn skipped(attrs: &[Attribute]) -> syn::Result<bool> {
    let mut skip = false;
    for attr in attrs {
        if !attr.path().is_ident("reflect") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                skip = true;
                Ok(())
            } else {
                Err(meta.error("unrecognized `reflect` option; expected `skip`"))
            }
        })?;
    }
    Ok(skip)
}
