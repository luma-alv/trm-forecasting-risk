"""Small optional Streamlit dashboard."""
from pathlib import Path


def main():
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError("install the 'app' extra to run the dashboard") from exc
    from trm_forecast.data.ingestion import load_trm
    from trm_forecast.models.baseline import forecast_last
    st.title("TRM Forecasting & Risk")
    path = st.text_input("TRM CSV path", "data/raw/trm.csv")
    horizon = st.slider("Forecast horizon", 1, 10, 5)
    if Path(path).exists():
        data = load_trm(path)
        st.line_chart(data["trm"])
        st.dataframe(forecast_last(data["trm"], horizon))
    else:
        st.info("Provide a local TRM CSV to begin.")


if __name__ == "__main__":
    main()
