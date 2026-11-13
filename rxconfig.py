import reflex as rx

config = rx.Config(
    app_name="dag_studio",
    plugins=[
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(
                appearance="light", accent_color="blue", gray_color="slate", radius="small"
            )
        )
    ],
    disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
)
