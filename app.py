import streamlit as st
import requests
import datetime
import pandas as pd
import re 
import google.generativeai as genai 
import os 
import time 
import altair as alt 


OPENWEATHER_API_KEY = os.getenv("WEATHER_API")
OPENWEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"

GEMINI_API_KEY = os.getenv("GEMINI_API")

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash') 
else:
    model = None 

# --- 熱中症指数(WBGT)計算関数 ---
def calculate_wbgt(temp_celsius, humidity_percent):
    if temp_celsius is None or humidity_percent is None:
        return None
    wbgt_approx = 0.735 * temp_celsius + 0.057 * humidity_percent - 2.82
    return max(0, wbgt_approx) 

# --- 水分補給量の計算ロジック ---
def calculate_base_water_intake(age, gender, weight_kg):
    if not all([age, gender, weight_kg]):
        return 0
    base_ml = 0
    if age < 18:
        base_ml = weight_kg * 40 
    elif age >= 65:
        base_ml = weight_kg * 30 
    else:
        base_ml = weight_kg * 35 
    return base_ml

def calculate_activity_water_loss(activity_type, duration_minutes):
    if activity_type == "ウォーキング":
        return duration_minutes * 5 
    elif activity_type == "ランニング":
        return duration_minutes * 10 
    elif activity_type == "サイクリング":
        return duration_minutes * 8 
    return 0

# --- Gemini API を使用した体調からのアドバイス生成関数 ---
@st.cache_data(show_spinner="Gemini AIがアドバイスを生成中...") 
def get_health_advice_from_gemini(mood_text, user_profile):
    if model is None:
        return "Gemini AIが利用できません。APIキーが正しく設定されているか確認してください。"

    if not mood_text:
        return "体調を入力してください。"

    prompt = f"""あなたは熱中症予防アプリ「HydroCare」の水分補給トレーナーです。
以下のユーザーの体調に関する記述を読み、体調が悪化しないように水分補給に関して具体的で優しいアドバイスを日本語で50文字程度で簡潔に提供してください。
医療行為ではなく、一般的な健康アドバイスの範囲でお願いします。
もしユーザーの年齢が{user_profile['age']}歳、体重が{user_profile['weight_kg']}kgの場合、それも考慮に入れてください。
---
ユーザーの体調: "{mood_text}"
---
アドバイス:
"""
    try:
        response = model.generate_content(prompt)
        if response._chunks: 
            return response.text 
        else:
            return "Gemini AI: 不適切な内容の可能性があり、アドバイスの生成がブロックされました。"
    except Exception as e:
        st.error(f"Gemini AIからのアドバイス取得中にエラーが発生しました: {e}")
        return "現在、AIからのアドバイス取得に問題が発生しています。しばらくしてから再度お試しください。"

# --- 新機能: Gemini API を使用した水分補給履歴のインサイト分析 ---
@st.cache_data(show_spinner="Gemini AIが水分補給習慣を分析中...")
def get_water_intake_insight_from_gemini(water_log, user_profile, base_daily_target_ml):
    if model is None:
        return "Gemini AIが利用できません。APIキーが正しく設定されているか確認してください。"
    
    if not water_log:
        return "まだ水分補給の記録がないため、分析できません。数日間の記録を付けてみましょう。"

    log_strings = []
    recent_logs = water_log[-10:] if len(water_log) > 10 else water_log
    for entry in recent_logs:
        log_time = pd.to_datetime(entry['time']).strftime('%Y年%m月%d日 %H時%M分')
        log_strings.append(f"- {log_time} に {entry['type']} を {entry['amount_ml']}ml 摂取")

    log_text = "\n".join(log_strings)

    prompt = f"""あなたは熱中症予防アプリ「HydroCare」の水分補給トレーナーです。
以下のユーザーの水分補給履歴、プロフィール、1日の基本目標水分量に基づいて、水分補給の習慣に関する具体的な「インサイト（洞察）」と「改善提案」を日本語で簡潔に100文字程度で提供してください。
医療行為ではなく、一般的な健康管理アドバイスの範囲でお願いします。

---
ユーザープロフィール:
年齢: {user_profile['age']}歳
体重: {user_profile['weight_kg']}kg
1日の基本目標水分量: {base_daily_target_ml}ml

最近の水分補給履歴（日付、時間、種類、量）:
{log_text}
---
分析と提案:
"""
    try:
        response = model.generate_content(prompt)
        if response._chunks:
            return response.text
        else:
            return "Gemini AI: 不適切な内容の可能性があり、分析の生成がブロックされました。"
    except Exception as e:
        st.error(f"Gemini AIからの分析取得中にエラーが発生しました: {e}")
        return "現在、AIからの分析取得に問題が発生しています。しばらくしてから再度お試しください。"


# --- 新機能: 次の水分補給時刻を計算 ---
def calculate_next_intake_time(last_intake_time, interval_minutes):
    if last_intake_time is None:
        return None 
    
    time_since_last = datetime.datetime.now() - last_intake_time
    remaining_seconds = (interval_minutes * 60) - time_since_last.total_seconds()
    
    if remaining_seconds <= 0:
        return "補給時間です！"
    else:
        minutes = int(remaining_seconds // 60)
        seconds = int(remaining_seconds % 60)
        return f"あと {minutes:02d}分 {seconds:02d}秒"

# --- 新機能: 日ごとの水分摂取量を集計 ---
def calculate_daily_summary(water_log, base_daily_target_ml):
    if not water_log:
        today = datetime.date.today()
        seven_days_ago = today - datetime.timedelta(days=6)
        date_range = pd.date_range(start=seven_days_ago, end=today, freq='D')
        df_empty = pd.DataFrame({'Date': date_range})
        df_empty['Total_ML'] = 0
        df_empty['Target_ML'] = base_daily_target_ml
        return df_empty

    df = pd.DataFrame(water_log)
    df['date'] = pd.to_datetime(df['time']).dt.date
    
    daily_summary_actual = df.groupby('date')['amount_ml'].sum().reset_index()
    daily_summary_actual.columns = ['Date', 'Total_ML']
    daily_summary_actual['Date'] = pd.to_datetime(daily_summary_actual['Date'])
    
    today = datetime.date.today()
    seven_days_ago = today - datetime.timedelta(days=6)
    date_range = pd.date_range(start=seven_days_ago, end=today, freq='D')
    
    full_df = pd.DataFrame({'Date': date_range})
    full_df = pd.merge(full_df, daily_summary_actual, on='Date', how='left').fillna(0)
    full_df['Total_ML'] = full_df['Total_ML'].astype(int) 

    full_df['Target_ML'] = base_daily_target_ml
    
    return full_df.sort_values('Date')


# --- メインアプリケーション関数 ---
def main_app():
    st.set_page_config(
        page_title="HydroCare - 熱中症予防アプリ", # アプリ名
        page_icon="💧",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("HydroCare") # アプリ名
    st.markdown("**あなたの水分チャージをスマートにお知らせ！夏を最高に楽しむための相棒アプリ！**")
    st.markdown("---")

    # --- セッションステートの初期化 (メインアプリ用) ---
    # ウォークスルーから遷移した場合でもセッションステートが引き継がれる
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'age': None,
            'gender': None,
            'weight_kg': None
        }
    if 'water_log' not in st.session_state:
        st.session_state.water_log = [] 
    if 'daily_target_ml' not in st.session_state:
        st.session_state.daily_target_ml = 0
    if 'total_consumed_ml' not in st.session_state:
        st.session_state.total_consumed_ml = 0
    if 'last_water_intake_time' not in st.session_state:
        st.session_state.last_water_intake_time = None
    if 'city_name' not in st.session_state:
        st.session_state.city_name = "Tokyo" 
    if 'reminder_interval_minutes' not in st.session_state:
        st.session_state.reminder_interval_minutes = 60 

    # --- サイドバーナビゲーション ---
    with st.sidebar:
        st.header("メニュー")
        page = st.radio(
            "表示するページを選択してください", 
            ["ホーム", "水分を記録", "摂取ログ", "天気とアクティビティ", "AIヘルスケア", "マイ設定"]
        )
        
        st.markdown("---")
        st.subheader("現在のユーザー情報")
        if st.session_state.user_profile['age'] and st.session_state.user_profile['weight_kg']:
            st.write(f"**年齢**: {st.session_state.user_profile['age']}歳")
            st.write(f"**体重**: {st.session_state.user_profile['weight_kg']} kg")
            st.write(f"**基本目標**: {st.session_state.daily_target_ml / 1000:.1f} L")
        else:
            st.info("プロフィールが未設定です。")


    # --- コンテンツ表示 ---

    # 1. 今日のサマリー 
    if page == "ホーム":
        st.header("🏠 ホーム")
        st.markdown("今日の水分補給状況と、次の補給タイミングを確認しましょう。")
        
        base_daily_target_ml = st.session_state.daily_target_ml
        
        if base_daily_target_ml == 0:
            st.warning("⚠️ **マイ設定**ページで年齢と体重を入力して、目標水分摂取量を計算してください。")
        else:
            st.info(f"**目標水分摂取量 (基本):** {base_daily_target_ml / 1000:.1f} リットル")

            current_consumed_ml = sum(entry['amount_ml'] for entry in st.session_state.water_log if pd.to_datetime(entry['time']).date() == datetime.date.today())
            st.session_state.total_consumed_ml = current_consumed_ml
            st.info(f"**本日摂取した水分量:** {current_consumed_ml / 1000:.1f} リットル")

            progress_percentage = (current_consumed_ml / base_daily_target_ml) * 100 if base_daily_target_ml > 0 else 0
            st.progress(min(int(progress_percentage), 100), text=f"目標達成度: {progress_percentage:.1f}%")

            if progress_percentage >= 100:
                st.balloons()
                st.success("素晴らしい！本日の水分補給目標を達成しました！")
            
            st.markdown("---")
            # 次の水分補給リマインダー表示
            st.subheader("⏰ 次の水分補給推奨時刻")
            next_intake_display = calculate_next_intake_time(st.session_state.last_water_intake_time, st.session_state.reminder_interval_minutes)
            if next_intake_display == "補給時間です！":
                st.warning("⏰ **水分補給の時間です！**")
            elif next_intake_display:
                st.info(f"次の水分補給まで: **{next_intake_display}**")
            else:
                st.info("水分補給の記録がまだありません。最初の補給を記録すると、次の推奨時刻が表示されます。")
            
            if isinstance(next_intake_display, str) and "あと" in next_intake_display:
                time.sleep(1) 
                st.rerun() 

            st.markdown("---")
            st.subheader("💧 あなたへの水分補給推奨")
            recommended_intake_ml = base_daily_target_ml - current_consumed_ml 
            if recommended_intake_ml < 0:
                recommended_intake_ml = 0 

            st.metric(label="推奨される水分補給量 (本日残り)", value=f"{recommended_intake_ml / 1000:.2f} リットル")
            st.write("この量は、あなたの基本情報と本日の摂取量に基づいています。活動量は「環境と活動量」ページで入力してください。")


    # 2. 水分を記録 
    elif page == "水分を記録":
        st.header("📝 水分を記録")
        st.write("飲んだ飲み物の種類と量を記録して、目標達成を目指しましょう！")

        drink_type = st.selectbox(
            "飲み物の種類",
            ["水", "お茶", "スポーツドリンク", "ジュース", "コーヒー", "その他"],
            key="drink_type_selector"
        )

        col_cup, col_bottle, col_slider = st.columns(3)

        with col_cup:
            st.write("### コップ1杯")
            if st.button("150ml 記録", key="record_150ml_btn"):
                st.session_state.water_log.append({'time': datetime.datetime.now(), 'amount_ml': 150, 'type': drink_type})
                st.session_state.last_water_intake_time = datetime.datetime.now()
                st.success(f"{drink_type}を150ml 記録しました！")
        
        with col_bottle:
            st.write("### ペットボトル")
            if st.button("500ml 記録", key="record_500ml_btn"):
                st.session_state.water_log.append({'time': datetime.datetime.now(), 'amount_ml': 500, 'type': drink_type})
                st.session_state.last_water_intake_time = datetime.datetime.now()
                st.success(f"{drink_type}を500ml 記録しました！")

        with col_slider:
            st.write("### カスタム量")
            custom_amount_ml_slider = st.slider(
                "飲んだ量 (ml)", 
                min_value=0, 
                max_value=1000, 
                value=250, 
                step=50,
                format="%d ml",
                key="custom_ml_slider"
            )
            if st.button(f"{custom_amount_ml_slider}ml を記録", key="custom_record_button"): 
                if custom_amount_ml_slider > 0:
                    st.session_state.water_log.append({'time': datetime.datetime.now(), 'amount_ml': custom_amount_ml_slider, 'type': drink_type})
                    st.session_state.last_water_intake_time = datetime.datetime.now()
                    st.success(f"{drink_type}を{custom_amount_ml_slider}ml 記録しました！")
                else:
                    st.warning("記録する量を0より大きく設定してください。")

        st.markdown("---")
        st.subheader("今日の水分補給履歴")
        if st.session_state.water_log:
            today_logs = [log for log in st.session_state.water_log if pd.to_datetime(log['time']).date() == datetime.date.today()]
            if today_logs:
                df_log = pd.DataFrame(today_logs)
                df_log['time'] = df_log['time'].dt.strftime('%H:%M:%S')
                st.dataframe(df_log.rename(columns={'time': '時刻', 'amount_ml': '摂取量 (ml)', 'type': '種類'}), use_container_width=True)
            else:
                st.info("まだ本日の水分補給記録はありません。")
        else:
            st.info("まだ水分補給記録はありません。")
        

    # 3. 摂取トレンド 
    elif page == "摂取ログ":
        st.header("📈 摂取ログ")
        st.markdown("過去の水分摂取傾向をグラフで確認し、習慣を改善しましょう。")

        base_daily_target_ml = st.session_state.daily_target_ml
        
        st.subheader("週間水分摂取ログ")
        df_daily_summary = calculate_daily_summary(st.session_state.water_log, base_daily_target_ml)
        
        # グラフのY軸最大値設定 (ml単位)
        dynamic_y_max = max(base_daily_target_ml * 1.5, 3000) if base_daily_target_ml > 0 else 3000
        
        if base_daily_target_ml == 0:
            st.warning("⚠️ **マイ設定**ページで年齢と体重を入力すると、目標線も表示されたグラフを見ることができます。")
            if not df_daily_summary.empty and not all(df_daily_summary['Total_ML'] == 0):
                # Altairチャートを作成
                chart = alt.Chart(df_daily_summary).mark_line().encode(
                    x=alt.X('Date:T', title='日付'),
                    y=alt.Y('Total_ML:Q', title='摂取量 (ml)', scale=alt.Scale(domain=[0, dynamic_y_max]))
                ).properties(
                    title='過去7日間の水分摂取量の推移'
                )
                st.altair_chart(chart, use_container_width=True) 
                st.write("過去7日間の水分摂取量の推移です。")
            else:
                st.info("まだ水分補給の記録が少ないため、トレンドは表示できません。数日間記録を続けると表示されます。")
        else:
            if not df_daily_summary.empty and not all(df_daily_summary['Total_ML'] == 0):
                # Altairチャートを作成
                chart = alt.Chart(df_daily_summary).mark_line().encode(
                    x=alt.X('Date:T', title='日付'),
                    y=alt.Y('value:Q', title='摂取量/目標量 (ml)', scale=alt.Scale(domain=[0, dynamic_y_max])), 
                    color=alt.Color('variable:N', title='項目', legend=alt.Legend(title="項目")) 
                ).transform_fold( 
                    ['Total_ML', 'Target_ML'],
                    as_=['variable', 'value']
                ).properties(
                    title='過去7日間の水分摂取量の推移'
                )
                st.altair_chart(chart, use_container_width=True) 
                st.write("過去7日間の水分摂取量の推移です。（青線: 摂取量, オレンジ線: 目標量）")
            else:
                st.info("まだ水分補給の記録が少ないため、トレンドは表示できません。数日間記録を続けると表示されます。")


    # 4. 環境と活動量 (旧環境と活動)
    elif page == "天気とアクティビティ":
        st.header("⛅ 天気")
        st.markdown("現在の気象情報と活動量から、水分補給の必要性を判断しましょう。")

        city_name_disabled = not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY"
        st.session_state.city_name = st.text_input(
            "現在の都市名を入力してください (例: Tokyo)", 
            value=st.session_state.city_name, 
            disabled=city_name_disabled
        )
        if st.button("天気情報を取得", disabled=city_name_disabled):
            if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY":
                st.error("OpenWeatherMap APIキーが設定されていません。コード内の 'OPENWEATHER_API_KEY' を置き換えてください。")
            else:
                try:
                    params = {
                        'q': st.session_state.city_name,
                        'appid': OPENWEATHER_API_KEY,
                        'units': 'metric', 
                        'lang': 'ja'
                    }
                    response = requests.get(OPENWEATHER_URL, params=params)
                    response.raise_for_status() 
                    weather_data = response.json()

                    main_data = weather_data.get('main', {})
                    temp = main_data.get('temp')
                    humidity = main_data.get('humidity')
                    description = weather_data.get('weather', [{}])[0].get('description', '不明')

                    st.write(f"**場所:** {st.session_state.city_name}")
                    if temp is not None and humidity is not None:
                        st.write(f"**気温:** {temp}°C")
                        st.write(f"**湿度:** {humidity}%")
                        st.write(f"**天気:** {description}")

                        wbgt = calculate_wbgt(temp, humidity)
                        if wbgt is not None:
                            st.write(f"**暑さ指数 (簡易WBGT):** {wbgt:.1f}°C")
                            if wbgt >= 31:
                                st.error("🚨 **危険レベル！** 厳重警戒が必要です。不要不急の外出を避け、涼しい場所で過ごし、こまめな水分補給を心がけてください。")
                            elif wbgt >= 28:
                                st.warning("⚠️ **厳重警戒！** 熱中症の危険が高まっています。運動は原則中止し、適切な水分補給を。")
                            elif wbgt >= 25:
                                st.warning("🟠 **警戒！** 熱中症になる危険性があります。積極的に水分補給を。")
                            else:
                                st.info("🟢 **注意！** 熱中症に注意し、適宜水分補給をしましょう。")
                        else:
                            st.warning("気温または湿度が取得できなかったため、WBGTは計算できませんでした。")
                    else:
                        st.warning("気温または湿度の情報が取得できませんでした。")

                except requests.exceptions.RequestException as e:
                    st.error(f"天気情報の取得に失敗しました: {e}")
                except KeyError:
                    st.error("天気データの形式が不正です。")
                except Exception as e:
                    st.error(f"予期せぬエラーが発生しました: {e}")
        
        st.markdown("---")
        st.header("🏃 アクティビティ")
        st.write("今日の活動量を入力して、必要な追加水分量を計算しましょう。")

        activity_type = st.selectbox("活動の種類", ["選択してください", "ウォーキング", "ランニング", "サイクリング"], key="activity_type_selector")
        duration_minutes = st.number_input("活動時間 (分)", min_value=0, value=0, key="duration_minutes_input")

        submitted_activity = st.button("活動量を更新", key="submit_activity_button")

        additional_water_needed_ml = 0
        if submitted_activity: 
            if activity_type != "選択してください" and duration_minutes > 0:
                additional_water_needed_ml = calculate_activity_water_loss(activity_type, duration_minutes)
                st.info(f"**{activity_type}を{duration_minutes}分行った場合、約 {additional_water_needed_ml} ml の追加水分が必要です。**")
                st.success("活動量を更新しました！")
            else:
                st.warning("活動の種類と時間を入力してください。")

        if st.session_state.daily_target_ml > 0:
            total_recommended_today = st.session_state.daily_target_ml + additional_water_needed_ml
            st.subheader("本日全体の推奨水分摂取量 (活動量考慮)")
            st.metric(label="合計推奨量", value=f"{total_recommended_today / 1000:.2f} リットル")
            st.write("この量は、あなたの基本情報と入力した活動量に基づいて計算されています。")
        else:
            st.warning("推奨水分摂取量を計算するには、まず「マイ設定」ページで年齢と体重を入力してください。")


    # 5. AIヘルスケア (旧AIアシスタント)
    elif page == "AIヘルスケア":
        st.header("✨ AIヘルスケア")
        st.markdown("AIを活用して、よりパーソナルな水分補給サポートを受けましょう！")

        st.markdown("---")

        # 気分・体調からの推奨調整
        st.subheader("🧠 気分・体調からのアドバイス")
        mood_text_disabled = (GEMINI_API_KEY == "YOUR_GEMINI_API_KEY")
        mood_input_text = st.text_area(
            "今日の気分や体調を教えてください（例: 少し疲れています、頭痛がします、元気です）", 
            key="mood_text_area",
            disabled=mood_text_disabled,
            placeholder="例: 少しだるいです、よく寝られませんでした。"
        )
        if st.button("AIアドバイスを得る", key="get_gemini_advice_button", disabled=mood_text_disabled):
            if mood_input_text:
                if st.session_state.user_profile['age'] is None or st.session_state.user_profile['weight_kg'] is None:
                    st.warning("より的確なアドバイスのため、**マイ設定**ページで年齢と体重を入力してください。")
                    temp_user_profile = {'age': 25, 'weight_kg': 60} 
                    advice = get_health_advice_from_gemini(mood_input_text, temp_user_profile)
                else:
                    advice = get_health_advice_from_gemini(mood_input_text, st.session_state.user_profile)
                
                if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                    st.error("Gemini APIキーが設定されていません。コード内の 'GEMINI_API_KEY' を置き換えてください。")
                else:
                    st.info(f"**AIからのアドバイス:** {advice}")
            else:
                st.warning("気分や体調を入力してください。")
        elif mood_text_disabled:
            st.error("Gemini APIキーが設定されていません。コード内の 'GEMINI_API_KEY' を置き換えてください。")

        st.markdown("---")

        # 水分補給履歴のインサイト分析
        st.subheader("📊 水分補給履歴のインサイト分析")
        st.write("過去の水分補給記録から、あなたの習慣に関する洞察と改善提案を得ましょう。")

        insight_disabled = mood_text_disabled 
        if st.button("インサイトを得る", key="get_insight_button", disabled=insight_disabled):
            if st.session_state.user_profile['age'] is None or st.session_state.user_profile['weight_kg'] is None:
                st.warning("より的確な分析のため、**マイ設定**ページで年齢と体重を入力してください。")
                temp_user_profile = {'age': 25, 'weight_kg': 60} 
                insight = get_water_intake_insight_from_gemini(
                    st.session_state.water_log, temp_user_profile, st.session_state.daily_target_ml
                )
            else:
                insight = get_water_intake_insight_from_gemini(
                    st.session_state.water_log, st.session_state.user_profile, st.session_state.daily_target_ml
                )
            
            if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                st.error("Gemini APIキーが設定されていません。コード内の 'GEMINI_API_KEY' を置き換えてください。")
            else:
                st.info(f"**AIからの分析:** {insight}")
        elif insight_disabled:
            st.error("Gemini APIキーが設定されていません。コード内の 'GEMINI_API_KEY' を置き換えてください。")


    # 6. マイ設定 (旧設定)
    elif page == "マイ設定":
        st.header("⚙️ マイ設定")
        st.markdown("アプリの基本設定やリマインダーを調整しましょう。")

        # プロフィール設定をここに移動
        st.subheader("👤 プロフィール設定")
        with st.form("profile_form_settings"): # フォーム名を変更
            age = st.number_input("年齢", min_value=1, max_value=120, value=st.session_state.user_profile['age'] if st.session_state.user_profile['age'] else 25, key="age_input_settings")
            gender = st.radio("性別", ("男性", "女性", "その他"), index=("男性", "女性", "その他").index(st.session_state.user_profile['gender']) if st.session_state.user_profile['gender'] else 0, key="gender_radio_settings")
            weight_kg = st.number_input("体重 (kg)", min_value=1.0, max_value=200.0, value=st.session_state.user_profile['weight_kg'] if st.session_state.user_profile['weight_kg'] else 60.0, key="weight_input_settings")
            
            submitted_profile_settings = st.form_submit_button("プロフィールを更新") 
            if submitted_profile_settings:
                st.session_state.user_profile['age'] = age
                st.session_state.user_profile['gender'] = gender
                st.session_state.user_profile['weight_kg'] = weight_kg
                st.success("プロフィールを更新しました！")
                
                st.session_state.daily_target_ml = calculate_base_water_intake(
                    st.session_state.user_profile['age'], 
                    st.session_state.user_profile['gender'], 
                    st.session_state.user_profile['weight_kg']
                )
        
        st.markdown("---")
        # リマインダー間隔設定をここに移動
        st.subheader("⏰ 水分補給リマインダー設定")
        with st.form("reminder_settings_form"): # リマインダー設定もフォームに含める
            st.session_state.reminder_interval_minutes = st.slider(
                "通知間隔 (分)",
                min_value=15,
                max_value=180,
                value=st.session_state.reminder_interval_minutes,
                step=15,
                format="%d分ごと",
                key="reminder_slider_settings"
            )
            st.write(f"現在の設定: **{st.session_state.reminder_interval_minutes}分** ごとに水分補給を推奨します。")
            submitted_reminder_settings = st.form_submit_button("リマインダー設定を更新")
            if submitted_reminder_settings:
                st.success("リマインダー設定を更新しました！")


    st.markdown("---")
    st.caption("© 2023 HydroCare. 熱中症予防をサポートします。")


# --- ウォークスルーロジック ---
# セッションステートでウォークスルーの完了状態とステップを管理
if 'walkthrough_completed' not in st.session_state:
    st.session_state.walkthrough_completed = False
if 'walkthrough_step' not in st.session_state:
    st.session_state.walkthrough_step = 0

# ウォークスルー画面の定義 (main_app関数の外に移動)
walkthrough_steps = [
    {
        "image": "data/Gemini_Generated_Image_66lomq66lomq66lo.png",
        "title": "いつでも水分チャージをスマートに！",
        "text": "プッシュ通知をONにすると、あなたに最適なタイミングで水分補給をお知らせします。"
    },
    {
        "image": "data/スクリーンショット_23-7-2025_02950_localhost.jpeg",
        "title": "活動や天気に合わせてパーソナル提案",
        "text": "あなたの活動レベルや天気・湿度から、今日必要な水分量をリアルタイムで自動計算します。"
    },
    {
        "image": "data/スクリーンショット_22-7-2025_235353_localhost.jpeg",
        "title": "記録も分析もAIにお任せ！",
        "text": "飲んだ量をタップで簡単記録。AIがあなたの水分補給習慣を分析して、健康維持をサポートします！"
    }
]

# アプリの開始点
if not st.session_state.walkthrough_completed:
    # ウォークスルー画面の描画
    walkthrough_placeholder = st.empty() 

    with walkthrough_placeholder.container():
        st.title("HydroCare へようこそ！")
        st.markdown("---")

        current_step_data = walkthrough_steps[st.session_state.walkthrough_step]

        # use_column_width を use_container_width に修正
        st.image(current_step_data["image"], use_container_width=True) 
        st.markdown(f"## {current_step_data['title']}")
        st.markdown(f"#### {current_step_data['text']}")
        
        st.markdown("---")
        
        # 進捗バー
        st.progress((st.session_state.walkthrough_step + 1) / len(walkthrough_steps))

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.session_state.walkthrough_step > 0:
                if st.button("戻る", key="walkthrough_back"):
                    st.session_state.walkthrough_step -= 1
                    st.rerun() 
        
        with col2:
            if st.session_state.walkthrough_step < len(walkthrough_steps) - 1:
                if st.button("次へ", key="walkthrough_next"):
                    st.session_state.walkthrough_step += 1
                    st.rerun() 
            else:
                if st.button("始める", key="walkthrough_start_app"):
                    st.session_state.walkthrough_completed = True
                    st.rerun() 
else:
    # ウォークスルー完了後、メインアプリを表示
    main_app()
